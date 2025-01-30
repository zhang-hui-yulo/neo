#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

#include "detail/data.h"


template <typename Config>
__global__ void __launch_bounds__((decltype(cute::size(typename Config::MMA{}))::value), 1)
gemm_multi_stage(void* Dptr, const void* Aptr, const void* Bptr, int m, int n,
    int k) {
    using namespace cute;

    using T = typename Config::T;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;

    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;

    extern __shared__ T shm_data[];

    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    // use Tensor notation to represent device pointer + dimension
    Tensor A = make_tensor(make_gmem_ptr((const T*)Aptr), make_shape(m, k),
        make_stride(k, Int<1>{}));  // (M, K)
    Tensor B = make_tensor(make_gmem_ptr((const T*)Bptr), make_shape(n, k),
        make_stride(k, Int<1>{}));  // (N, K)
    Tensor D = make_tensor(make_gmem_ptr((T*)Dptr), make_shape(m, n),
        make_stride(n, Int<1>{}));  // (M, N)

    // slice the tensor to small one which is used for current thread block.
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
        make_coord(iy, _));  // (kTileM, kTileK, k)
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
        make_coord(ix, _));  // (kTileN, kTileK, k)
    Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
        make_coord(iy, ix));  // (kTileM, kTileN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm),
        SmemLayoutA{});  // (kTileM, kTileK)
    auto sB = make_tensor(make_smem_ptr(Bshm),
        SmemLayoutB{});  // (kTileN, kTileK)

    // gmem -ld 128-> shm -ld 128-> reg
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy =
        g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K)
    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    auto tArA_copy = make_fragment_like(tAsA_copy);  // (CPY, CPY_M, CPY_K)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy =
        g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K)
    auto tBrB_copy = make_fragment_like(tBsB_copy);  // (CPY, CPY_N, CPY_K)

    // Copy gmem to rmem for k_tile=0
    copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, _0{}), tArA_copy);
    copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, _0{}), tBrB_copy);

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    // method
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, _0{}));  // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, _0{}));  // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)

    // fill zero for accumulator
    clear(tCrD);

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // ? (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // ? (CPY, CPY_M, CPY_K)

    // Copy rmem to smem
    copy(tArA_copy, tAsA_copy);
    copy(tBrB_copy, tBsB_copy);

    __syncthreads();

    // Load A, B shmem->regs for k_block=0
    copy(tAsA(_, _, _0{}), tCrA_view(_, _, _0{}));
    copy(tBsB(_, _, _0{}), tCrB_view(_, _, _0{}));
    int K_TILE_MAX = size<2>(gA);
    int K_BLOCK_MAX = size<2>(tCrA);
    
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        // Pipeline the k-mode of the block registers
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Copy rmem to smem
                __syncthreads();
                copy(tArA_copy, tAsA_copy);
                copy(tBrB_copy, tBsB_copy);
                __syncthreads();
            }

            // Copy smem to rmem for k_block+1
            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(tAsA(_, _, k_block_next), tCrA_view(_, _, k_block_next));
            copy(tBsB(_, _, k_block_next), tCrB_view(_, _, k_block_next));
            if (k_block == 0) {
                // Copy gmem to rmem for k_tile+1
                int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
                copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, k_tile_next), tArA_copy);
                copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, k_tile_next), tBrB_copy);
            }

            // Thread-level register gemm for k_block
            gemm(tiled_mma, tCrD, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrD);
        }
    }

    // use less shared memory as a scratchpad tile to use large wide instuction
    // Dreg -> shm -> reg -> global
    auto sC = make_tensor(sA.data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s);  // pipe
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        // reg -> shm
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
            copy(tCrC_r2sx(_, i + j), t);

            copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

        CUTE_UNROLL
        // shm -> global
        for (int j = 0; j < step; ++j) {
            copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }

        __syncthreads();
    }
}


namespace config {

using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kSmemLayoutCBatch_ = 2,
          typename ComputeType = T_>
struct GemmConfig {
    using T = T_;

    // tile configuration
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 3;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
            make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtom{},
            make_shape(Int<kTileM>{}, Int<kTileK>{})));
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtom{},
            make_shape(Int<kTileN>{}, Int<kTileK>{})));

    using mma_op = GFX11_16x16x16_F16F16F16F16_TN;

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op = UniversalCopy<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
            make_layout(make_shape(Int<32>{}, Int<4>{}),
            make_stride(Int<4>{}, Int<1>{})),
            make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    // shared memory to register copy
    using s2r_copy_op = UniversalCopy<cute::uint128_t>;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // epilogue: register to global via shared memory
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
            make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
        "C shared memory request is large than A's one pipe");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<half_t>, T>;

    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
            make_layout(make_shape(Int<32>{}, Int<4>{}),
                make_stride(Int<4>{}, Int<1>{})),
            make_layout(make_shape(Int<1>{}, Int<8>{}))));

    static constexpr int kThreadNum = size(MMA{});
    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

    static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}  // namespace config


int main() {
    using T = cute::half_t;
    using namespace cute;

    srand(10086);

    // default;
    int M = 81920;
    int N = 256;
    int K = 256;

    int nt = 11;

    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 32;

    thrust::host_vector<T> Aptr_host(M * K);
    thrust::host_vector<T> Bptr_host(N * K);
    thrust::host_vector<T> Dptr_host(M * N, 0);

    thrust::host_vector<T> Dptr_host_blas(M * N, 0);

    auto tA = make_tensor(Aptr_host.data(), make_layout(make_shape(M, K), GenRowMajor{}));
    auto tB = make_tensor(Bptr_host.data(), make_layout(make_shape(N, K), GenRowMajor{}));

    cpu_rand_data(&tA);
    cpu_rand_data(&tB);

    thrust::device_vector<T> Aptr = Aptr_host;
    thrust::device_vector<T> Bptr = Bptr_host;
    thrust::device_vector<T> Dptr = Dptr_host;

    thrust::device_vector<T> Dptr_cublas = Dptr_host_blas;

    config::GemmConfig<T, kTileM, kTileN, kTileK> gemm_config;

    print(typename decltype(gemm_config)::MMA{});

    dim3 block = gemm_config.kThreadNum;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
        (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    int shm_size = gemm_config.kShmSize;

    half alpha = 1.f;
    half beta = 0.f;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    for (int it = 0; it < nt; ++it) {
        Dptr_cublas = Dptr_host_blas;
        hipblasStatus_t ret = hipblasHgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K,
            (hipblasHalf*)&alpha, (hipblasHalf*)Bptr.data().get(), K, (hipblasHalf*)Aptr.data().get(), K,
            (hipblasHalf*)&beta, (hipblasHalf*)Dptr_cublas.data().get(), N);
        if (ret != HIPBLAS_STATUS_SUCCESS) {
            printf("cublas err = %d, str = %s\n", ret, hipblasStatusToString(ret));
        }

        Dptr = Dptr_host;
        hipFuncSetAttribute((const void*)gemm_multi_stage<decltype(gemm_config)>,
            hipFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        
        gemm_multi_stage<decltype(gemm_config)>
            <<<grid, block, shm_size>>> (Dptr.data().get(), Aptr.data().get(), Bptr.data().get(), M, N, K);
    }

    Dptr_host = Dptr;
    Dptr_host_blas = Dptr_cublas;

    hipDeviceSynchronize();
    auto err = hipGetLastError();
    printf("block = (%d, %d), gird = (%d, %d), shm = %d\n", block.x, block.y,
        grid.x, grid.y, shm_size);

    if (err == hipSuccess) {
        printf("err = %d, str = %s\n", err, hipGetErrorString(err));
    } else {
        printf_fail("err = %d, str = %s\n", err, hipGetErrorString(err));
    }

    gpu_compare(Dptr.data().get(), Dptr_cublas.data().get(), M * N);

    auto tD_host = make_tensor(Dptr_host.data(), make_layout(make_shape(M, N), GenRowMajor{}));
    auto tD_host_blas =
        make_tensor(Dptr_host_blas.data(), make_layout(make_shape(M, N), GenRowMajor{}));

    auto tile = make_tile(min(8, M), min(8, N));
    auto t32x32 = local_tile(tD_host, tile, make_coord(0, 0));
    auto t32x32_blas = local_tile(tD_host_blas, tile, make_coord(0, 0));

    printf("M = %d, N = %d, K = %d\n", M, N, K);

    printf("our-impl:\n");
    print_tensor(t32x32);

    printf("cublas:\n");
    print_tensor(t32x32_blas);

    // measure the performance
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    constexpr int warmup = 100;
    constexpr int ntime = 1000;
    hipFuncSetAttribute((const void*)gemm_multi_stage<decltype(gemm_config)>,
        hipFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    // gemm warmup
    for (int i = 0; i < warmup; ++i) {
        gemm_multi_stage<decltype(gemm_config)>
            <<<grid, block, shm_size>>> (Dptr.data().get(), Aptr.data().get(), Bptr.data().get(), M, N, K);
    }

    float time_mma = 0;

    for (int i = 0; i < ntime; ++i) {
        hipEventRecord(start);

        gemm_multi_stage<decltype(gemm_config)>
            <<<grid, block, shm_size>>> (Dptr.data().get(), Aptr.data().get(), Bptr.data().get(), M, N, K);

        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float elapsed = 0;
        hipEventElapsedTime(&elapsed, start, stop);
        time_mma += elapsed;

        auto err = hipGetLastError();
        if (err != hipSuccess) {
            printf_fail("err = %d, str = %s\n", err, hipGetErrorString(err));
        }
    }

    // cublas warmup
    for (int i = 0; i < warmup; ++i) {
        hipblasStatus_t ret = hipblasHgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K,
            (hipblasHalf*)&alpha, (hipblasHalf*)Bptr.data().get(), K, (hipblasHalf*)Aptr.data().get(), K,
            (hipblasHalf*)&beta, (hipblasHalf*)Dptr_cublas.data().get(), N);
    }

    float time_hipblas = 0;

    for (int i = 0; i < ntime; ++i) {
        hipEventRecord(start);

        hipblasStatus_t ret = hipblasHgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K,
            (hipblasHalf*)&alpha, (hipblasHalf*)Bptr.data().get(), K, (hipblasHalf*)Aptr.data().get(), K,
            (hipblasHalf*)&beta, (hipblasHalf*)Dptr_cublas.data().get(), N);

        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float elapsed = 0;
        hipEventElapsedTime(&elapsed, start, stop);
        time_hipblas += elapsed;

        auto err = hipGetLastError();
        if (err != hipSuccess) {
            printf_fail("err = %d, str = %s\n", err, hipGetErrorString(err));
        }
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipblasDestroy(handle);

    printf_ok("check ok, hip mma = %f, hipblas = %f\n", time_mma / ntime, time_hipblas / ntime);
}
