#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>


template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T* Cptr, const T* Aptr, const T* Bptr, int m, int n, int k) {
    using namespace cute;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), LayoutRight{});
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), LayoutRight{});
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), LayoutRight{});

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)

    clear(tCrC);

    int num_tile_k = size<2>(gA);

    #pragma unroll
    for (int itile = 0; itile < num_tile_k; ++itile) {
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC);
}


template <typename T>
void gen_rand_data(T* data, int n) {
    for (int i = 0; i < n; ++i) {
        float v = (rand() % 200 - 100) * 0.01;
        data[i] = v;
    }
}


int main() {
    srand(10086);

    using T = cute::half_t;
    using namespace cute;

    int m = 81920, n = 256, k = 256;
    // Define a tensor shape with dynamic extents (m, k)
    auto tensor_shapeA = make_shape(m, k);
    // Define a tensor shape with dynamic extents (n, k)
    auto tensor_shapeB = make_shape(n, k);
    // Define a tensor shape with dynamic extents (m, n)
    auto tensor_shapeC = make_shape(m, n);

    //
    // Allocate and initialize
    //

    thrust::host_vector<T> h_A(size(tensor_shapeA));
    thrust::host_vector<T> h_B(size(tensor_shapeB));

    gen_rand_data(h_A.data(), m * k);
    gen_rand_data(h_B.data(), n * k);

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C(size(tensor_shapeC));

    using mma_op = GFX11_16x16x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, \
        make_layout(Shape<_2, _2, _1>{}),
        make_layout(Shape<_1, _1, _1>{})));
    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 32;

    dim3 block(size(MMA{}));
    dim3 grid(n / kTileN, m / kTileM);
    for (int i = 0; i < 100; ++i) {
        gemm_simple<T, kTileM, kTileN, kTileK, MMA> << <grid, block >> > (d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);
    }
    hipDeviceSynchronize();
    auto err = hipGetLastError();
    printf("err = %d, str = %s\n", err, hipGetErrorString(err));

    thrust::device_vector<T> d_C_blas(size(tensor_shapeC));

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    half alpha = half(1.f);
    half beta = half(0.f);
    for (int i = 0; i < 100; ++i) {
        hipblasStatus_t ret = hipblasHgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
            n, m, k,
            (hipblasHalf*)&alpha,
            (hipblasHalf*)d_B.data().get(), k,
            (hipblasHalf*)d_A.data().get(), k,
            (hipblasHalf*)&beta,
            (hipblasHalf*)d_C_blas.data().get(), n);
        if (ret != HIPBLAS_STATUS_SUCCESS) {
            printf("blas err = %d, str = %s\n", ret, hipblasStatusToString(ret));
        }
    }

    hipDeviceSynchronize();
    err = hipGetLastError();
    printf("err = %d, str = %s\n", err, hipGetErrorString(err));

    thrust::host_vector<T> h_C = d_C;
    thrust::host_vector<T> h_C_blas = d_C_blas;

    float threshold = 0.1;
    for (int i = 0; i < m * n; ++i) {
        float v1 = h_C[i];
        float v2 = h_C_blas[i];
        if (fabs(v2 - v1) > threshold) {
            printf("v1 = %f, v2 = %f\n", v1, v2);
        }
    }

    Tensor tensor_C = make_tensor(h_C.data(), make_shape(m, n), make_stride(n, 1));
    Tensor tensor_C_cublas = make_tensor(h_C_blas.data(), make_shape(m, n), make_stride(n, 1));

    auto tile = make_tile(8, 8);
    auto coor = make_coord(0, 0);
    Tensor tc1 = local_tile(tensor_C, tile, coor);
    Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

    print_tensor(tc1);
    print_tensor(tc1_cublas);
}
