#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

using namespace kittens;

#define ATTN_B 16 // batch size
#define ATTN_H 16 // number of heads
#define ATTN_N 4096 // sequence length
#define ATTN_D 64 // dimension
#define BLOCK_SIZE 32 // block size

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_QKVO = gl<bf16, -1, -1, -1, ATTN_D>;
struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ((ATTN_N / BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 20000; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void attend_ker(const attn_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, ATTN_D> (&k_smem)[2] = al.allocate<st_bf<BLOCK_SIZE, ATTN_D>, 2>();
    st_bf<BLOCK_SIZE, ATTN_D> (&v_smem)[2] = al.allocate<st_bf<BLOCK_SIZE, ATTN_D>, 2>();
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_tile_idx = blockIdx.z;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();

    const float scale_factor = 1.0f / sqrt(ATTN_D);

    // Initialize all of the register tiles.
    rt_bf<BLOCK_SIZE, ATTN_D> q_reg, k_reg; // Q and K are both row layout.
    rt_bf<BLOCK_SIZE, ATTN_D, ducks::rt_layout::col> v_reg;
    rt_fl<BLOCK_SIZE, ATTN_D, ducks::rt_layout::col> o_reg;
    rt_fl<BLOCK_SIZE, ATTN_D, ducks::rt_layout::accumulator> o_reg_next;
    rt_fl<BLOCK_SIZE, ATTN_D, ducks::rt_layout::col> o_reg_next_col; // attention tile, in float, for the mma_AB.
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::accumulator> att_block;
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::row> att_block_row;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::row> att_block_row_bf16;
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col>::col_vec max_vec_last, max_vec, max_vec_new, norm_vec_last, norm_vec, norm_vec_new; 

    // 5. Given i = blockIdx.x, load Q_i from global to registers. Set O_i = 0, l_i = 0, m_i = -inf.
    zero(o_reg);
    zero(norm_vec_last);
    zero(norm_vec);
    zero(norm_vec_new);
    neg_infty(max_vec_last);
    neg_infty(max_vec);
    neg_infty(max_vec_new);

    int num_tiles = ATTN_N / BLOCK_SIZE;
    int num_kv_per_iter = 2;

    bool valid_q_tile = (tile_idx < num_tiles);
    if (valid_q_tile) {
        load(q_reg, g.Qg, {batch_idx, head_idx, tile_idx, 0});
    }

    int tic = 0, toc = 1;
    if (warpid() < 2) {
        load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<BLOCK_SIZE,ATTN_D>>, NUM_THREADS / 2>(
        g.Kg, {batch_idx, head_idx, 0, 0}, k_smem[tic]);
    }
    if (warpid() >= 2) {
        load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<BLOCK_SIZE,ATTN_D>>, NUM_THREADS / 2, WARP_THREADS * 2>(
        g.Vg, {batch_idx, head_idx, 0, 0}, v_smem[tic]);
    }
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();        
    
    for (int j = 0; j < num_tiles; j++, tic^=1, toc^=1) {

        if (warpid() < 2 && j < num_tiles - 1) {
            load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<BLOCK_SIZE,ATTN_D>>, NUM_THREADS / 2>(
                g.Kg, {batch_idx, head_idx, j + 1, 0}, k_smem[toc]);
        }
        if (warpid() >= 2 && j < num_tiles - 1) {
            load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<BLOCK_SIZE,ATTN_D>>, NUM_THREADS / 2, WARP_THREADS * 2>(
                g.Vg, {batch_idx, head_idx, j + 1, 0}, v_smem[toc]);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __syncthreads();    

        zero(att_block);
        zero(o_reg_next);

        // Load K and V
        load_lds_reg(k_reg, k_smem[tic]);
        load_lds_reg(v_reg, v_smem[tic]);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        //  Compute Q_i @ K_j.T 
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        att_block_row = swap_layout_inplace(att_block);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // Softmax
        mul(att_block_row, att_block_row, scale_factor);
        row_max(max_vec, att_block_row);  
        sub_row(att_block_row, att_block_row, max_vec); 
        exp(att_block_row, att_block_row); 

        // Normalization
        row_sum(norm_vec, att_block_row);
        max(max_vec_new, max_vec_last, max_vec); 
        sub(max_vec_last, max_vec_last, max_vec_new); 
        exp(max_vec_last, max_vec_last);
        sub(max_vec, max_vec, max_vec_new);
        exp(max_vec, max_vec);
        mul(norm_vec_last, max_vec_last, norm_vec_last);
        mul(norm_vec, max_vec, norm_vec);
        add(norm_vec_new, norm_vec_last, norm_vec);

        // V multiplication
        mul_row(o_reg, o_reg, max_vec_last); 
        copy(att_block_row_bf16, att_block_row);
        __builtin_amdgcn_s_setprio(1);
        mma_AB(o_reg_next, att_block_row_bf16, v_reg, o_reg_next);
        o_reg_next_col = swap_layout_inplace(o_reg_next);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        
        // Outputs and running normalization
        mul_row(o_reg_next_col, o_reg_next_col, max_vec);
        add(o_reg, o_reg, o_reg_next_col);
        copy(max_vec_last, max_vec_new); 
        copy(norm_vec_last, norm_vec_new);
        __builtin_amdgcn_s_barrier();

    }

    if (valid_q_tile) {
        div_row(o_reg, o_reg, norm_vec_last);
        store(g.Og, o_reg, {batch_idx, head_idx, tile_idx, 0});
    }
}


void dispatch_micro(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &attn_globals::Qg, &attn_globals::Kg, &attn_globals::Vg, &attn_globals::Og);
}