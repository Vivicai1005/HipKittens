#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_N = 32; // sequence length
constexpr int ATTN_D = 32; // dimension
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qkvo_tile_transposed = rt<T, D, BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=row_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;

template<int D> struct attn_bwd_combined_globals { 
    gl<bf16, -1, -1, -1, -1> Q, K, V, O;
    gl<float, -1, -1, -1, -1> dOg, dQg;
    gl<float, -1, -1, -1, -1> m_vec, l_vec;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z;  // This is the block index we're processing

    const float scale_factor = 1.0f / sqrt(D);

    // Register tiles
    qkvo_tile<D, bf16, row_l> qi_reg, ki_reg, vi_reg;
    qkvo_tile<D, bf16, col_l> ki_reg_col, vi_reg_col;
    qkvo_tile<D, float, row_l> dOi_reg, Oi_reg;

    // Load this block's data (block i)
    load(qi_reg,  g.Q,  {b,h,i,0});
    load(ki_reg,  g.K,  {b,h,i,0});
    load(vi_reg,  g.V,  {b,h,i,0});
    load(dOi_reg, g.dOg, {b,h,i,0});
    load(Oi_reg,  g.O,  {b,h,i,0});
    
    // Load statistics for block i
    typename attn_tile<D,float,accum_col_l>::col_vec mi_vec, li_vec;
    load(mi_vec, g.m_vec, {b,h,0,0});
    load(li_vec, g.l_vec, {b,h,0,0});

    int num_blocks = ATTN_N / BLOCK_SIZE;
    
    for (int j = 0; j < num_blocks; ++j) {
        
        // Load K_j and V_j
        qkvo_tile<D, bf16, row_l> kj_reg, vj_reg;
        load(kj_reg, g.K, {b,h,j,0});
        load(vj_reg, g.V, {b,h,j,0});

        // S_ij = (Q_i K_j^T) * scale
        attn_tile<D,float,accum_col_l> S_ij; 
        zero(S_ij);
        mma_ABt(S_ij, qi_reg, kj_reg, S_ij);
        mul(S_ij, S_ij, scale_factor);

        // P_ij = exp(S_ij - m_i) / l_i
        sub_row(S_ij, S_ij, mi_vec);
        exp(S_ij, S_ij);
        // div_row(S_ij, S_ij, li_vec);
    
        store(g.dQg, S_ij, {b,h,i,j});
    }

    // Store results for block i
    // store(g.dQg, mi_vec, {b,h,i,0});
}

template<int D>
void dispatch_bwd_combined(attn_bwd_combined_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

// Add to PYBIND11_MODULE:
PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";

    // Combined backward pass kernel
    py::bind_function<dispatch_bwd_combined<ATTN_D>>(m, "dispatch_bwd_combined", 
        &attn_bwd_combined_globals<ATTN_D>::Q, 
        &attn_bwd_combined_globals<ATTN_D>::K, 
        &attn_bwd_combined_globals<ATTN_D>::V, 
        &attn_bwd_combined_globals<ATTN_D>::O, 
        &attn_bwd_combined_globals<ATTN_D>::dOg, 
        &attn_bwd_combined_globals<ATTN_D>::dQg,
        &attn_bwd_combined_globals<ATTN_D>::m_vec, 
        &attn_bwd_combined_globals<ATTN_D>::l_vec
    );
}

