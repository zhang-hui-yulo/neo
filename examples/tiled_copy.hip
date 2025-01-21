/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>


/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
    using namespace cute;

    // Slice the tiled tensors
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

    // Construct a partitioning of the tile among threads with the given thread arrangement.

    // Concept:                         Tensor  ThrLayout       ThrIndex
    Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
    Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_tensor to try to match the layout of thr_tile_S
    Tensor fragment = make_tensor_like(thr_tile_S);               // (ThrValM, ThrValN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(thr_tile_S, fragment);
    copy(fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
    using namespace cute;

    // Slice the tensors to obtain a view into each tile.
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

    // Construct a Tensor corresponding to each thread's slice.
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
    Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}

int main() {
    //
    // Given a 2D shape, perform an efficient copy
    //

    using namespace cute;
    using Element = float;

    // Define a tensor shape with dynamic extents (m, n)
    auto tensor_shape = make_shape(256, 512);

    //
    // Allocate and initialize
    //

    thrust::host_vector<Element> h_S(size(tensor_shape));
    thrust::host_vector<Element> h_D(size(tensor_shape));

    for (size_t i = 0; i < h_S.size(); ++i) {
        h_S[i] = static_cast<Element>(i);
        h_D[i] = Element{};
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    //
    // Make tensors
    //

    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));

    //
    // Tile tensors
    //

    // Define a statically sized block (M, N).
    // Note, by convention, capital letters are used to represent static modes.
    auto block_shape = make_shape(Int<128>{}, Int<64>{});

    if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
        std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
        return -1;
    }
    // Equivalent check to the above
    if (not evenly_divides(tensor_shape, block_shape)) {
        std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
        return -1;
    }

    // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
    // shape, and modes (m', n') correspond to the number of tiles.
    //
    // These will be used to determine the CUDA kernel grid dimensions.
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

    // Construct a TiledCopy with a specific access pattern.
    //   This version uses a
    //   (1) Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc),
    //   (2) Layout-of-Values that each thread will access.

    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx

    // Value arrangement per thread
    Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4,1) -> val_idx

    // Define `AccessType` which controls the size of the actual memory access instruction.
    using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element)* size(val_layout)>>;     // A very specific access width copy instruction
    //using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
    //using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

    // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
    using Atom = Copy_Atom<CopyOp, Element>;

    // Construct tiled copy, a tiling of copy atoms.
    //
    // Note, this assumes the vector and thread layouts are aligned with contigous data
    // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
    // reads. Alternative value layouts are also possible, though incompatible layouts
    // will result in compile time errors.
    TiledCopy tiled_copy = make_tiled_copy(Atom{},             // Access strategy
        thr_layout,         // thread layout (e.g. 32x4 Col-Major)
        val_layout);        // value layout (e.g. 4x1)

    //
    // Determine grid and block dimensions
    //

    dim3 gridDim(size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(thr_layout));

    //
    // Launch the kernel
    //
    copy_kernel_vectorized << < gridDim, blockDim >> > (
        tiled_tensor_S,
        tiled_tensor_D,
        tiled_copy);

    hipError_t result = hipDeviceSynchronize();
    if (result != hipSuccess) {
        std::cerr << "CUDA Runtime error: " << hipGetErrorString(result) << std::endl;
        return -1;
    }

    //
    // Verify
    //

    h_D = d_D;

    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    for (size_t i = 0; i < h_D.size(); ++i) {
        if (h_S[i] != h_D[i]) {
            std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

            if (++errors >= kErrorLimit) {
                std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
                return -1;
            }
        }
    }

    std::cout << "Success." << std::endl;

    return 0;
}
