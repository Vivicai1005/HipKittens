#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_N = 32; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qkvo_tile_transposed = rt<T, D, BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=row_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;

template<int D> struct attn_globals { 
    gl<bf16, -1, -1, -1, -1> Qg, Kg, Vg, Og, dOg, dQg;
    gl<bf16, -1, -1, -1, 1> m_vec, l_vec;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_dq_ker(const attn_globals<D> g) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z;

    const float scale_factor = 1.0f / sqrt(D);

    // tiles
    qkvo_tile<D, bf16, row_l> q_reg, k_reg, v_reg;
    qkvo_tile<D, bf16, col_l> k_reg_col;
    qkvo_tile<D, bf16, row_l> dO_reg, O_reg;
    qkvo_tile<D, float, accum_col_l> dQ_acc; 
    zero(dQ_acc);

    // load Q_i, dO_i, O_i, and stats (m,l)
    load(q_reg,  g.Qg,  {b,h,i,0});
    load(dO_reg, g.dOg, {b,h,i,0});
    load(O_reg,  g.Og,  {b,h,i,0});
    typename attn_tile<D,float,col_l>::col_vec m_vec, l_vec;
    load(m_vec, g.m_vec, {b,h,i,0});
    load(l_vec, g.l_vec, {b,h,i,0});

    // Δ_i = row_sum(dO ⊙ O) 
    qkvo_tile<D, float, row_l> tmp_float;
    qkvo_tile<D, float, row_l> dO_float, O_float;
    
    // Convert to float for computation
    copy(dO_float, dO_reg);
    copy(O_float, O_reg);
    
    mul(tmp_float, dO_float, O_float);
    attn_tile<D,float,row_l>::col_vec delta_vec;
    row_sum(delta_vec, tmp_float); 

    /* TODO: Fix vector shapes. */
    one(l_vec);
    one(m_vec);
    one(delta_vec);

    int num_blocks = ATTN_N/BLOCK_SIZE;
    for (int j = 0; j < num_blocks; ++j) {
        load(k_reg, g.Kg, {b,h,j,0});
        load(v_reg, g.Vg, {b,h,j,0});

        // Convert k_reg to col layout
        swap_layout(k_reg_col, k_reg);

        // S_ij = (Q_i K_j^T)*scale
        attn_tile<D,float,accum_col_l> S; 
        zero(S);
        mma_ABt(S, q_reg, k_reg, S);
        mul(S, S, scale_factor);

        // P_ij = exp(S - m)/l
        sub_row(S, S, m_vec);
        exp(S, S);
        div_row(S, S, l_vec);
        attn_tile<D,float,accum_col_l> P; 
        copy(P, S);

        // dOVt = dO_i V_j^T
        attn_tile<D,float,accum_col_l> dOVt; 
        zero(dOVt);
        mma_ABt(dOVt, dO_reg, v_reg, dOVt);

        // dS = P ⊙ (dOVt - Delta)
        sub_col(dOVt, dOVt, delta_vec);
        mul(dOVt, dOVt, P);

        // dQ += dS K_j * scale
        qkvo_tile<D,float,accum_col_l> dQ_blk; 
        zero(dQ_blk);
        
        // Convert dOVt to row layout
        attn_tile<D,float,row_l> dOVt_row;
        swap_layout(dOVt_row, dOVt);
        // Convert to bf16 for MMA operation
        attn_tile<D,bf16,row_l> dOVt_bf16_row;
        copy(dOVt_bf16_row, dOVt_row);
        
        mma_AB(dQ_blk, dOVt_bf16_row, k_reg_col, dQ_blk);
        mul(dQ_blk, dQ_blk, scale_factor);
        add(dQ_acc, dQ_acc, dQ_blk);
    }

    // store dQ (bf16)
    qkvo_tile<D,bf16,accum_col_l> dQ_reg; 
    copy(dQ_reg, dQ_acc);
    store(g.dQg, dQ_reg, {b,h,i,0});
}


template<int D>
struct bwd_dkv_globals {
    gl<bf16, -1, -1, -1, -1> Qg, Kg, Vg, Og, dOg, dKg, dVg;
    gl<bf16, -1, -1, -1, 1> m_vec, l_vec;   

    gl<bf16, -1, -1, -1, -1> test;
    dim3 grid()  const { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() const { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() const { return MAX_SHARED_MEMORY-32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_bwd_dkv_ker(const bwd_dkv_globals<D> g) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int j = blockIdx.z;
    const float scale = 1.0f / sqrtf((float)D);

    qkvo_tile<D,bf16,row_l> k_reg;
    qkvo_tile<D,bf16,row_l> v_reg;
    qkvo_tile_transposed<D, bf16> v_reg_transposed;
    load(k_reg, g.Kg, {b,h,j,0});
    load(v_reg, g.Vg, {b,h,j,0});
    __syncthreads();
    __builtin_amdgcn_s_waitcnt(0);

    // accumulators for the outputs
    qkvo_tile<D,float,accum_col_l> dV_acc; 
    zero(dV_acc);
    qkvo_tile<D,float,accum_col_l> dK_acc; 
    zero(dK_acc);

    int num_blocks = ATTN_N/BLOCK_SIZE;
    for (int i = 0; i < num_blocks; ++i) {

        qkvo_tile<D,bf16, row_l> q_reg;
        qkvo_tile<D,float,row_l> dO_f, O_f;
        typename attn_tile<D,float,col_l>::col_vec m_vec, l_vec;
        
        load(q_reg,  g.Qg,    {b,h,i,0});
        load(dO_f,   g.dOg,   {b,h,i,0});
        load(O_f,    g.Og,    {b,h,i,0});
        load(m_vec,  g.m_vec, {b,h,i,0});
        load(l_vec,  g.l_vec, {b,h,i,0});
        one(l_vec);
        one(m_vec);
        
        // P_ij
        attn_tile<D,float,accum_col_l> S; 
        zero(S);
        mma_ABt(S, q_reg, k_reg, S);
        mul(S, S, scale);
        sub_row(S, S, m_vec);
        exp(S, S);
        div_row(S, S, l_vec); 
        attn_tile<D,float,accum_col_l> P; 
        copy(P, S);
        
        // dV += P^T dO_i
        attn_tile<D,bf16,accum_col_l> P_bf16; 
        copy(P_bf16, P);
        attn_tile<D,bf16,col_l> P_bf16_col;
        swap_layout(P_bf16_col, P_bf16);
        
        qkvo_tile<D,float,accum_col_l> dV_blk; 
        zero(dV_blk);
        qkvo_tile<D,bf16,row_l> dO_bf16;
        copy(dO_bf16, dO_f);
        qkvo_tile<D,bf16,col_l> dO_bf16_col;
        swap_layout(dO_bf16_col, dO_bf16);
        mma_AtB(dV_blk, P_bf16_col, dO_bf16_col, dV_blk); 
        add(dV_acc, dV_acc, dV_blk);

        // Delta_i
        qkvo_tile<D,float,row_l> tmp;
        mul(tmp, dO_f, O_f);
        attn_tile<D,float,row_l>::col_vec delta_vec;
        row_sum(delta_vec, tmp);
        one(delta_vec);
        
        // dS = P ⊙ (dO_i V_j^T − Delta)
        attn_tile<D,float,accum_col_l> dOVt; zero(dOVt);
        mma_ABt(dOVt, dO_bf16, v_reg, dOVt); 
        sub_col(dOVt, dOVt, delta_vec);
        mul(dOVt, dOVt, P);
        
        // dK += dS^T Q_i * scale
        attn_tile<D,bf16,accum_col_l> dS_bf16; 
        copy(dS_bf16, dOVt);
        auto dS_bf16_row = swap_layout_inplace<col_l>(dS_bf16);
        qkvo_tile<D,float,accum_col_l> dK_blk; 
        zero(dK_blk);
        qkvo_tile<D,bf16,col_l> q_bf16_col;
        swap_layout(q_bf16_col, q_reg);
        mma_AtB(dK_blk, dS_bf16_row, q_bf16_col, dK_blk);
        mul(dK_blk, dK_blk, scale);
        add(dK_acc, dK_acc, dK_blk);
    }

    // store dK,dV (bf16)
    qkvo_tile<D,bf16,accum_col_l> dV_bf16; 
    copy(dV_bf16, dV_acc);
    store(g.dVg, dV_bf16, {b,h,j,0});
    qkvo_tile<D,bf16,accum_col_l> dK_bf16; 
    copy(dK_bf16, dK_acc);
    store(g.dKg, dK_bf16, {b,h,j,0});
}



/*******************************************
* Dispatch functions
*******************************************/

template<int D>
void dispatch_micro(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_dq_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_dq_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}


template<int D>
void dispatch_bwd_dkv(bwd_dkv_globals<D> g){
  hipFuncSetAttribute((void*)attend_bwd_dkv_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());
  attend_bwd_dkv_ker<D><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
  hipDeviceSynchronize();
}


PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", 
        &attn_globals<ATTN_D>::Qg, 
        &attn_globals<ATTN_D>::Kg, 
        &attn_globals<ATTN_D>::Vg, 
        &attn_globals<ATTN_D>::Og, 
        &attn_globals<ATTN_D>::dOg, 
        &attn_globals<ATTN_D>::dQg,
        &attn_globals<ATTN_D>::m_vec, 
        &attn_globals<ATTN_D>::l_vec
    );

    py::bind_function<dispatch_bwd_dkv<ATTN_D>>(m, "dispatch_bwd_dkv", 
        &bwd_dkv_globals<ATTN_D>::Qg, 
        &bwd_dkv_globals<ATTN_D>::Kg, 
        &bwd_dkv_globals<ATTN_D>::Vg, 
        &bwd_dkv_globals<ATTN_D>::Og, 
        &bwd_dkv_globals<ATTN_D>::dOg, 
        &bwd_dkv_globals<ATTN_D>::dKg, 
        &bwd_dkv_globals<ATTN_D>::dVg, 
        &bwd_dkv_globals<ATTN_D>::m_vec, 
        &bwd_dkv_globals<ATTN_D>::l_vec,
        &bwd_dkv_globals<ATTN_D>::test
    );
}

