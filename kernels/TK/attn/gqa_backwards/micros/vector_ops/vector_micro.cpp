#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int b = 1;
constexpr int h = 1;
constexpr int n = 32;
constexpr int d = 32;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

struct micro_globals {
    gl<float, -1, -1, -1, -1> in;
    gl<float, -1, -1, -1, -1> in_vec;
    gl<float, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-2048; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    rt_fl<n, d, accum_col_l> tile_accum;
    rt_fl<n, d, accum_col_l>::row_vec vec;
    load(tile_accum, g.in, {0, 0, 0, 0});
    load(vec, g.in_vec, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();
    // one(tile_accum);

    sub_col(tile_accum, tile_accum, vec);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();

    store(g.out, tile_accum, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in, &micro_globals::in_vec, &micro_globals::out);
}

