#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 32
#define K 32
#define N 32

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    using ST_A = st_bf<32, 32, st_16x32_s>;
    using ST_B = st_bf<32, 32, st_16x32_s>;
    ST_A (&As) = al.allocate<ST_A>();
    ST_B (&Bs) = al.allocate<ST_B>();

    using A_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<128, 135>>, 4>; // 8 registers - v[128:135]
    using B_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<136, 143>>, 4>; // 8 registers - v[136:143]
    using C_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<144, 159>>, 4>; // 16 registers - v[144:159]

    ducks::rt::clobber<A_ranges>();
    ducks::rt::clobber<B_ranges>();
    ducks::rt::clobber<C_ranges>();

    rt<bf16, 32, 32, row_l, rt_16x32_s, A_ranges> A_tile;
    rt<bf16, 32, 32, row_l, rt_16x32_s, B_ranges> B_tile;
    rt<float, 32, 32, col_l, rt_16x16_s, C_ranges> C_tile;

    // global to shared loads
    load(As, g.a, {0, 0, 0, 0});
    load(Bs, g.b, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // shared to registers
    load(A_tile, As);
    load(B_tile, Bs);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // mma
    mma_ABt(C_tile, A_tile, B_tile);

    // registers to global
    store(g.c, C_tile, {0, 0, 0, 0}, {0, 0, 0, 0});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    // py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
