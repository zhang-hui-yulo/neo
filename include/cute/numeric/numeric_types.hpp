/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>          // CUTE_HOST_DEVICE
#include <cute/numeric/int.hpp>     // cute::int2_t, cute::int4_t, etc

#if !defined(__HIP_PLATFORM_AMD__)
#include <cutlass/numeric_size.h>   // cutlass::sizeof_bits
#include <cutlass/numeric_types.h>  // cutlass::float_e4m3_t, cutlass::float_e5m2_t, etc
#else
#include <hip/hip_fp16.h>           // hip half type
#include <hip/hip_bfloat16.h>       // hip_bfloat16
#endif

namespace cute {

#if !defined(__HIP_PLATFORM_AMD__)
template <typename T>
struct sizeof_bits : public cutlass::sizeof_bits<T> {};
#else
/// Defines the size of an element in bits
template <typename T>
struct sizeof_bits {
  static constexpr int value = int(sizeof(T) * 8);
};

template <typename T>
struct sizeof_bits<T const>: sizeof_bits<T> {};

template <>
struct sizeof_bits<void> {
  static constexpr int value = 0;
};
#endif

// DO NOT change auto to int, sizeof_bits<sparse_elem> use integral_ratio instead of int 
template <class T>
static constexpr auto sizeof_bits_v = sizeof_bits<T>::value;

#if !defined(__HIP_PLATFORM_AMD__)
using cutlass::bits_to_bytes;
using cutlass::bytes_to_bits;

using cutlass::is_subbyte;
#else
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the number of bytes required to hold a specified number of bits
template <class R = int, class T>
CUTE_HOST_DEVICE
constexpr
R
bits_to_bytes(T bits) {
  return (R(bits) + R(7)) / R(8);
}

/// Returns the number of bits required to hold a specified number of bytes
template <class R = int, class T>
CUTE_HOST_DEVICE
constexpr
R
bytes_to_bits(T bytes) {
  return R(bytes) * R(8);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
struct is_subbyte {
  static constexpr bool value = sizeof_bits<T>::value < 8;
};

template <class T>
struct is_subbyte<T const> : is_subbyte<T> {};
#endif

template <class T>
static constexpr auto is_subbyte_v = is_subbyte<T>::value;

#if !defined(__HIP_PLATFORM_AMD__)
using cutlass::half_t;
using cutlass::bfloat16_t;

using cutlass::tfloat32_t;

// Umbrella floating-point 8-bit data type : type_erased_dynamic_float8_t
// This umbrella datatype can be enabled when a user provides a specific
// datatype in runtime argument list.
using cutlass::type_erased_dynamic_float8_t;
using cutlass::float_e4m3_t;
using cutlass::float_e5m2_t;

using cutlass::uint1b_t;
using cutlass::int2b_t;
using cutlass::uint2b_t;
using cutlass::int4b_t;
using cutlass::uint4b_t;
using cutlass::bin1_t;
#else
using half_t = half;
using bfloat16_t = hip_bfloat16;
#endif


//
// Print utility
//

CUTE_HOST_DEVICE
void
print(half_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(bfloat16_t a) {
  printf("%f", static_cast<float>(a));
}


#if !defined(__HIP_PLATFORM_AMD__)
CUTE_HOST_DEVICE
void
print(tfloat32_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(float_e4m3_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(float_e5m2_t a) {
  printf("%f", static_cast<float>(a));
}
#endif

CUTE_HOST_DEVICE void
pretty_print(bfloat16_t v) {
  printf("%*.2f", 8, float(v));
}

CUTE_HOST_DEVICE void
pretty_print(half_t v) {
  printf("%*.2f", 8, float(v));
}

#if !defined(__HIP_PLATFORM_AMD__)
CUTE_HOST_DEVICE void
pretty_print(tfloat32_t v) {
  printf("%*.2e", 10, static_cast<float>(v));
}

CUTE_HOST_DEVICE void
pretty_print(float_e4m3_t t) {
  printf("%*.2f", 8, static_cast<float>(t));
}

CUTE_HOST_DEVICE void
pretty_print(float_e5m2_t t) {
  printf("%*.2f", 8, static_cast<float>(t));
}
#endif

} // namespace cute
