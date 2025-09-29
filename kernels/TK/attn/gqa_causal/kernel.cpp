#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_H_KV = 1; // number of heads for key and value
constexpr int ATTN_N = 512; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

template<int D, typename T=bf16, typename L=row_l> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L>;
template<int D, typename T=bf16, typename L=row_l> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=accum_col_l> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L>;


/**********************************************************/


template<ducks::rt::accumulator_col_layout RT>  // RT = rt<float, KV_BLOCK_SIZE, Q_BLOCK_SIZE, accum_col_l>
__device__ inline void mask_causal_kv_q_accum_col(
    RT &dst,
    int q_abs,
    int k_abs,
    const typename base_types::packing<typename RT::dtype>::unpacked_type &neg_inf
) {
    if (k_abs > q_abs + (Q_BLOCK_SIZE - 1)) { neg_infty(dst); return; } // whole future tile
    if (k_abs + KV_BLOCK_SIZE - 1 <= q_abs) { return; }            // whole past tile

    const int lane = laneid();
    const int col  = lane & 31;   // query column
    const int ro   = lane >> 5;   // row-block selector

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {      // i steps rows by 32
        const int base32 = i * 32;
        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int rbase = base32 + ii*8 + ro*4;  // row within KV tile

                auto &d0 = dst.tiles[i][j].data[ii*2];
                auto &d1 = dst.tiles[i][j].data[ii*2 + 1];

                if (k_abs + (rbase + 0) > q_abs + col) d0.x = neg_inf;
                if (k_abs + (rbase + 1) > q_abs + col) d0.y = neg_inf;
                if (k_abs + (rbase + 2) > q_abs + col) d1.x = neg_inf;
                if (k_abs + (rbase + 3) > q_abs + col) d1.y = neg_inf;
            }
        }
    }
}

template<typename AT>
__device__ inline void mask_kv_tile(AT &A, int q_tile, int k_tile) {
    const int q_abs = q_tile * Q_BLOCK_SIZE;
    const int k_abs = k_tile * KV_BLOCK_SIZE;
    mask_causal_kv_q_accum_col(A, q_abs, k_abs,
        kittens::base_types::constants<float>::neg_infty());
}


/**********************************************************/


template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    gl<float, -1, -1, -1, -1> L_vec;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col>, 2>();
    
    const int head_idx = (blockIdx.x % 8) * 8 + (blockIdx.x / 8);
    // const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int GROUP_SIZE = ATTN_H / ATTN_H_KV;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    int k_idx_buf0 = -1, k_idx_buf1 = -1;
    int k_curr_idx = -1; 


    int causal = true;
    int num_tiles;
    if (causal) {
        // tile_idx is the actual position of this query tile
        int q_end_pos = (tile_idx + 1) * Q_BLOCK_SIZE;
        num_tiles = (q_end_pos + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
        num_tiles = min(num_tiles, ATTN_N / KV_BLOCK_SIZE);
    } else {
        num_tiles = ATTN_N / KV_BLOCK_SIZE;
    }
    int condition = (laneid() == 0 && blockIdx.x == 0 && blockIdx.z == 0);
    if (condition) {
        printf("warpid: %d, block_tile_idx: %d, num_tiles: %d\n", warpid(), block_tile_idx, num_tiles);
    }

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;

    kv_tile<D, bf16, accum_col_l> v_reg;
    qo_tile_transposed<D, float, accum_col_l> o_reg; // Output tile.
    attn_tile<D, float, accum_col_l> att_block[2]; // attention tile, in float.
    attn_tile<D, bf16, accum_col_l> att_block_bf16;
    typename attn_tile<D, float, accum_col_l>::row_vec max_vec, norm_vec, max_vec_prev;

    using T = typename st_bf<KV_BLOCK_SIZE, ATTN_D>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);

    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K); 
    k_idx_buf0 = 0;
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    qo_tile<D, float> q_reg_fl;
    load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    copy(q_reg, q_reg_fl);
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec_prev);

    // All warps then collaboratively load in the first slice of V (V0) and the second slice of K (K1) into shared memory
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K);
    k_idx_buf1 = 1; 
    // All warps then load in the first slice of K (K0)
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V);
    load(k_reg, k_smem[0]);
    k_curr_idx = k_idx_buf0; 
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Each warp performs QK0
    zero(att_block[0]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    if (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);

    // Each warp performs a partial softmax of QK0 (i.e. some of the online softmax up until but not including the second exponential scaling of the attention block likely)
    col_max(max_vec, att_block[0]);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // All warps then load in the second slice of K (K1)
    if (num_tiles > 1) {
        load(k_reg, k_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        k_curr_idx = k_idx_buf1; 
    }
    // All warps then collaboratively load in the third slice of K (K2) into shared memory
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K);
    k_idx_buf0 = 2; 
    // All warps then collaboratively load in the second slice of V (V1) into shared memory 
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    if (num_tiles <= 3) {
        // First block doesn't have a previous max, so we just use the current max
        col_sum(norm_vec, att_block[0]);  // Sum the attention weights
        copy(att_block_bf16, att_block[0]);
        
        // Load V0 and accumulate
        load(v_reg, v_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
        
        // Save max_vec for next iteration
        copy(max_vec_prev, max_vec);
        
        if (num_tiles >= 2) {
            // Process block 1
            // QK1 computation (k_reg already loaded)
            zero(att_block[1]);
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
            if (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
            
            // Softmax update with proper rescaling
            col_max(max_vec, att_block[1]);  // Get new max
            sub(max_vec_prev, max_vec_prev, max_vec);  // old_max - new_max
            exp2(max_vec_prev, max_vec_prev);  // exp2(old_max - new_max)
            mul(norm_vec, norm_vec, max_vec_prev);
            mul_col(o_reg, o_reg, max_vec_prev);
            
            // Now process the new attention block
            sub_col(att_block[1], att_block[1], max_vec);
            exp2(att_block[1], att_block[1]);
            col_sum(norm_vec, att_block[1], norm_vec);  // Add to rescaled norm
            copy(att_block_bf16, att_block[1]);
            
            // Load V1 and accumulate
            load(v_reg, v_smem[1]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            
            // Update max_vec_prev for next iteration
            copy(max_vec_prev, max_vec);
        }
        
        if (num_tiles == 3) {
            // Load K2 from shared memory
            load(k_reg, k_smem[0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            k_curr_idx = k_idx_buf0;
            
            // QK2 computation
            zero(att_block[0]);
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
            if (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
            
            // Get new maximum
            col_max(max_vec, att_block[0]);
            sub(max_vec_prev, max_vec_prev, max_vec);
            exp2(max_vec_prev, max_vec_prev);
            mul(norm_vec, norm_vec, max_vec_prev);
            mul_col(o_reg, o_reg, max_vec_prev);
            
            // Process new attention block
            sub_col(att_block[0], att_block[0], max_vec);
            exp2(att_block[0], att_block[0]);
            col_sum(norm_vec, att_block[0], norm_vec);
            copy(att_block_bf16, att_block[0]);
            
            // Load V2 (need to load it first since it wasn't preloaded)
            load<1, false>(v_smem[0], g.Vg, {batch_idx, 2, head_idx_kv, 0});
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            
            load(v_reg, v_smem[0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            
            // Update for final normalization
            copy(max_vec_prev, max_vec);
        }
        
        // Normalize and store output
        div_col(o_reg, o_reg, norm_vec);
        
        qo_tile<D, float, accum_row_l> o_reg_transposed;
        swap_layout_and_transpose(o_reg_transposed, o_reg);
        store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});
        
        // Store log-sum-exp (using max_vec_prev which contains the final max)
        mul(max_vec_prev, max_vec_prev, 0.69314718056f);  // Convert to natural log
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec_prev);
        store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
        
        return;
    }

    // hot loop
    // #pragma unroll  // for some reason unroll makes it slower
    for (int j = 3; j < num_tiles - 1; j += 2) {
        // Cluster 0:
        //      QK1
        zero(att_block[1]);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
        if (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
        //      Finish softmax for QK0
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[0], norm_vec);
        copy(att_block_bf16, att_block[0]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1:
        //      Load K3 into shared 
        G::load<1, false>(k_smem[1], g.Kg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_K);
        k_idx_buf1 = j; 
        //      Load V0 into registers
        load(v_reg, v_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2:
        //      A0V0
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
        //      Partial softmax for QK1
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[1], max_vec);
        sub_col(att_block[1], att_block[1], max_vec);
        exp2(att_block[1], att_block[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3:
        //      Load V2 into shared
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V);
        //      Load K2 into registers
        load(k_reg, k_smem[0]);
        k_curr_idx = k_idx_buf0;
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4:
        //      QK2
        zero(att_block[0]);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
        if (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
        //      Finish softmax for QK1
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[1], norm_vec);
        copy(att_block_bf16, att_block[1]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5:
        //      Load K4 into shared
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, swizzled_offsets_K);
        k_idx_buf0 = j + 1;
        //      Load V1 into registers
        load(v_reg, v_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6:
        //      A1V1
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
        //      Partial softmax for QK2
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[0], max_vec);
        sub_col(att_block[0], att_block[0], max_vec);
        exp2(att_block[0], att_block[0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7:
        //      Load V3 into shared
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V);
        //      Load K3 into registers
        load(k_reg, k_smem[1]);
        k_curr_idx = k_idx_buf1;
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0:
    //      QK3
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    if (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
    //      Finish softmax for QK2
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);

    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1:
    //      Load K5 into shared
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K);
    //      Load V2 into registers
    load(v_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    //      A2V2
    __builtin_amdgcn_s_setprio(1);
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Partial softmax for QK3
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3:
    //      Load V4 into shared
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V);
    //      Load K4 into registers
    load(k_reg, k_smem[0]);
    k_curr_idx = k_idx_buf0;
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    //      QK4
    zero(att_block[0]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    if (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
    //      Finish softmax for QK3
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);

    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5:
    //      Load V3 into registers
    load(v_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6:
    //      A3V3
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Partial softmax for QK4
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[0], max_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7:
    //      Load V5 into shared
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V);
    //      Load K5 into registers
    load(k_reg, k_smem[1]);
    k_curr_idx = k_idx_buf1;
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    //      QK5
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    if (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
    //      Finish softmax for QK4
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev); 
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9:
    //      Load V4 into registers
    load(v_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10:
    //      A4V4
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Full softmax for QK5
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    sub(max_vec_prev, max_vec_prev, max_vec);
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11:
    //      Load V5 into registers
    load(v_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    //      A5V5
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    div_col(o_reg, o_reg, norm_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    qo_tile<D, float, accum_row_l> o_reg_transposed;
    swap_layout_and_transpose(o_reg_transposed, o_reg);
    store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});

    // multiply by ln(2)
    mul(max_vec, max_vec, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}

template<int D>
void dispatch_micro(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", 
        &attn_globals<ATTN_D>::Qg, 
        &attn_globals<ATTN_D>::Kg, 
        &attn_globals<ATTN_D>::Vg, 
        &attn_globals<ATTN_D>::Og,
        &attn_globals<ATTN_D>::L_vec
    );
}
