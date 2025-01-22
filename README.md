# neo
NVIDIA CuTe AMD porting based on CUTLASS 3.7.0

This is an experimental project to port CuTe to AMD GPU, hopefully it can reach peak performance.

Todo list:
- Add non native type support in numeric/int.hpp like uint1b_t, uint4b_t, uint128b_t
- clean up all CUTE_UNROLL as the limiation of clang
- add gfx gemm support
- port reed's cute samples
