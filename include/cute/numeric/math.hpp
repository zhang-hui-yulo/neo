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

#include <cute/config.hpp>            // CUTE_HOST_DEVICE
#include <cute/util/type_traits.hpp>  // __CUTE_REQUIRES

#if !defined(__HIP_PLATFORM_AMD__)
#include <cutlass/fast_math.h>
#endif

namespace cute
{
#if defined(__HIP_PLATFORM_AMD__)
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * log2 computation, what's the
 * difference between the below codes and
 * log2_up/down codes?
 */
template <typename value_t>
CUTE_HOST_DEVICE
constexpr
value_t clz(value_t x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x)
      return value_t(31 - i);
  }
  return value_t(32);
}

template <typename value_t>
CUTE_HOST_DEVICE
constexpr
value_t find_log2(value_t x) {
  int a = int(31 - clz(x));
  a += (x & (x - 1)) != 0;  // Round up, add 1 if not a power of 2.
  return a;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation.
///
/// This object precomputes two values used to accelerate the computation and is best used
/// when the divisor is a grid-invariant. In this case, it may be computed in host code and
/// marshalled along other kernel arguments using the 'Params' pattern.
///
/// Example:
///
///
///   int quotient, remainder, dividend, divisor;
///
///   FastDivmod divmod(divisor);
///
///   divmod(quotient, remainder, dividend);
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmod {
  using value_div_type = int;
  using value_mod_type = int64_t;
  int32_t divisor = 1;
  uint32_t multiplier = 0u;
  uint32_t shift_right = 0u;

  // Find quotient and remainder using device-side intrinsics
  CUTE_HOST_DEVICE
  void fast_divmod(int& quotient, int& remainder, int dividend) const {

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // Use IMUL.HI if divisor != 1, else simply copy the source.
    quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
    quotient = int((divisor != 1) ? int(((int64_t)dividend * multiplier) >> 32) >> shift_right : dividend);
#endif

    // The remainder.
    remainder = dividend - (quotient * divisor);
  }

  /// For long int input
  CUTE_HOST_DEVICE
  void fast_divmod(int& quotient, int64_t& remainder, int64_t dividend) const {

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // Use IMUL.HI if divisor != 1, else simply copy the source.
    quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
    quotient = int((divisor != 1) ? ((dividend * multiplier) >> 32) >> shift_right : dividend);
#endif
    // The remainder.
    remainder = dividend - (quotient * divisor);
  }


  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.

  constexpr FastDivmod() = default;

  CUTE_HOST_DEVICE
  FastDivmod(int divisor_): divisor(divisor_) {
    assert(divisor_ >= 0);
    if (divisor != 1) {
      unsigned int p = 31 + find_log2(divisor);
      unsigned m = unsigned(((1ull << p) + unsigned(divisor) - 1) / unsigned(divisor));

      multiplier = m;
      shift_right = p - 32;
    }
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTE_HOST_DEVICE
  void operator()(int &quotient, int &remainder, int dividend) const {
    fast_divmod(quotient, remainder, dividend);
  }

  /// Computes integer division using precomputed values. This is computationally
  /// inexpensive.
  CUTE_HOST_DEVICE
  int div(int dividend) const {
    int quotient, remainder;
    fast_divmod(quotient, remainder, dividend);
    return quotient;
  }

  /// Alias for `div` to match the interface of FastDivmodU64
  CUTE_HOST_DEVICE
  int divide(int dividend) const {
    return div(dividend);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  ///
  /// Simply returns the quotient
  CUTE_HOST_DEVICE
  int divmod(int &remainder, int dividend) const {
    int quotient;
    fast_divmod(quotient, remainder, dividend);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTE_HOST_DEVICE
  void operator()(int &quotient, int64_t &remainder, int64_t dividend) const {
    fast_divmod(quotient, remainder, dividend);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTE_HOST_DEVICE
  int divmod(int64_t &remainder, int64_t dividend) const {
    int quotient;
    fast_divmod(quotient, remainder, dividend);
    return quotient;
  }

  /// Returns the divisor when cast to integer
  CUTE_HOST_DEVICE
  operator int() const { return divisor; }

};
#endif

//
// Common Operations
//

template <class T, class U,
          __CUTE_REQUIRES(is_arithmetic<T>::value &&
                          is_arithmetic<U>::value)>
CUTE_HOST_DEVICE constexpr
auto
max(T const& t, U const& u) {
  return t < u ? u : t;
}

template <class T, class U,
          __CUTE_REQUIRES(is_arithmetic<T>::value &&
                          is_arithmetic<U>::value)>
CUTE_HOST_DEVICE constexpr
auto
min(T const& t, U const& u) {
  return t < u ? t : u;
}

template <class T,
          __CUTE_REQUIRES(is_arithmetic<T>::value)>
CUTE_HOST_DEVICE constexpr
auto
abs(T const& t) {
  if constexpr (is_signed<T>::value) {
    return t < T(0) ? -t : t;
  } else {
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

// Returns 1 if x > 0, -1 if x < 0, and 0 if x is zero.
template <class T,
          __CUTE_REQUIRES(is_arithmetic<T>::value)>
CUTE_HOST_DEVICE constexpr
int
signum(T const& x) {
  if constexpr (is_signed<T>::value) {
    return (T(0) < x) - (x < T(0));
  } else {
    return T(0) < x;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// C++17 <numeric> operations
//

// Greatest common divisor of two positive integers
template <class T, class U,
          __CUTE_REQUIRES(is_std_integral<T>::value &&
                          is_std_integral<U>::value)>
CUTE_HOST_DEVICE constexpr
cute::common_type_t<T, U>
gcd(T t, U u) {
  while (true) {
    if (t == 0) { return u; }
    u %= t;
    if (u == 0) { return t; }
    t %= u;
  }
}

// Least common multiple of two positive integers
template <class T, class U,
          __CUTE_REQUIRES(is_std_integral<T>::value &&
                          is_std_integral<U>::value)>
CUTE_HOST_DEVICE constexpr
cute::common_type_t<T, U>
lcm(T const& t, U const& u) {
  return (t / gcd(t,u)) * u;
}

//
// C++20 <bit> operations
//

// Checks if a number is an integral power of two
template <class T>
CUTE_HOST_DEVICE constexpr
bool
has_single_bit(T x) {
  return x != 0 && (x & (x - 1)) == 0;
}

// Smallest number of bits needed to represent the given value
//   For x == 0, this is 0
//   For x != 0, this is 1 + floor(log2(x))
// bit_width( 0b0000 ) = 0
// bit_width( 0b0001 ) = 1
// bit_width( 0b0010 ) = 2
// bit_width( 0b0011 ) = 2
// bit_width( 0b0100 ) = 3
// bit_width( 0b0101 ) = 3
// bit_width( 0b0110 ) = 3
// bit_width( 0b0111 ) = 3
template <class T>
CUTE_HOST_DEVICE constexpr
int
bit_width(T x) {
  static_assert(is_unsigned<T>::value, "Only to be used for unsigned types.");
  constexpr int N = (numeric_limits<T>::digits == 64 ? 6 :
                    (numeric_limits<T>::digits == 32 ? 5 :
                    (numeric_limits<T>::digits == 16 ? 4 :
                    (numeric_limits<T>::digits ==  8 ? 3 : (assert(false),0)))));
  T r = 0;
  for (int i = N - 1; i >= 0; --i) {
    T shift = (x > ((T(1) << (T(1) << i))-1)) << i;
    x >>= shift;
    r  |= shift;
  }
  return r + (x != 0);
}

// Smallest integral power of two not less than the given value
// bit_ceil( 0b00000000 ) = 0b00000001
// bit_ceil( 0b00000001 ) = 0b00000001
// bit_ceil( 0b00000010 ) = 0b00000010
// bit_ceil( 0b00000011 ) = 0b00000100
// bit_ceil( 0b00000100 ) = 0b00000100
// bit_ceil( 0b00000101 ) = 0b00001000
// bit_ceil( 0b00000110 ) = 0b00001000
// bit_ceil( 0b00000111 ) = 0b00001000
// bit_ceil( 0b00001000 ) = 0b00001000
// bit_ceil( 0b00001001 ) = 0b00010000
template <class T>
CUTE_HOST_DEVICE constexpr
T
bit_ceil(T x) {
  return x == 0 ? T(1) : (T(1) << bit_width(x - 1));
}

// Largest integral power of two not greater than the given value
// bit_floor( 0b00000000 ) = 0b00000000
// bit_floor( 0b00000001 ) = 0b00000001
// bit_floor( 0b00000010 ) = 0b00000010
// bit_floor( 0b00000011 ) = 0b00000010
// bit_floor( 0b00000100 ) = 0b00000100
// bit_floor( 0b00000101 ) = 0b00000100
// bit_floor( 0b00000110 ) = 0b00000100
// bit_floor( 0b00000111 ) = 0b00000100
// bit_floor( 0b00001000 ) = 0b00001000
// bit_floor( 0b00001001 ) = 0b00001000
template <class T>
CUTE_HOST_DEVICE constexpr
T
bit_floor(T x) {
  return x == 0 ? 0 : (T(1) << (bit_width(x) - 1));
}

template <class T>
CUTE_HOST_DEVICE constexpr T rotl(T x, int s);
template <class T>
CUTE_HOST_DEVICE constexpr T rotr(T x, int s);

// Computes the result of circular bitwise left-rotation
template <class T>
CUTE_HOST_DEVICE constexpr
T
rotl(T x, int s) {
  constexpr int N = numeric_limits<T>::digits;
  return static_cast<T>(s == 0 ? x : s > 0 ? (x << s) | (x >> (N - s)) : rotr(x, -s));
}

// Computes the result of circular bitwise right-rotation
template <class T>
CUTE_HOST_DEVICE constexpr
T
rotr(T x, int s) {
  constexpr int N = numeric_limits<T>::digits;
  return static_cast<T>(s == 0 ? x : s > 0 ? (x >> s) | (x << (N - s)) : rotl(x, -s));
}

// Counts the number of consecutive 0 bits, starting from the most significant bit
// countl_zero( 0b00000000 ) = 8
// countl_zero( 0b11111111 ) = 0
// countl_zero( 0b00011100 ) = 3
template <class T>
CUTE_HOST_DEVICE constexpr
int
countl_zero(T x) {
  return numeric_limits<T>::digits - bit_width(x);
}

// Counts the number of consecutive 1 bits, starting from the most significant bit
// countl_one( 0b00000000 ) = 0
// countl_one( 0b11111111 ) = 8
// countl_one( 0b11100011 ) = 3
template <class T>
CUTE_HOST_DEVICE constexpr
int
countl_one(T x) {
  return countl_zero(~x);
}

// Counts the number of consecutive 0 bits, starting from the least significant bit
// countr_zero( 0b00000000 ) = 8
// countr_zero( 0b11111111 ) = 0
// countr_zero( 0b00011100 ) = 2
template <class T>
CUTE_HOST_DEVICE constexpr
int
countr_zero(T x) {
  return x == 0 ? numeric_limits<T>::digits : bit_width(T(x & T(-x))) - 1;  // bit_width of the LSB
}

// Counts the number of consecutive 1 bits, starting from the least significant bit
// countr_one( 0b00000000 ) = 0
// countr_one( 0b11111111 ) = 8
// countr_one( 0b11100011 ) = 2
template <class T>
CUTE_HOST_DEVICE constexpr
int
countr_one(T x) {
  return countr_zero(~x);
}

// Counts the number of 1 bits in an unsigned integer
// popcount( 0b00000000 ) = 0
// popcount( 0b11111111 ) = 8
// popcount( 0b00011101 ) = 4
template <class T>
CUTE_HOST_DEVICE constexpr
int
popcount(T x) {
  int c = 0;
  while (x) {
    ++c;
    x &= x - 1; // clear the least significant bit set
  }
  return c;
}

//
// Custom operations
//

// Computes the result of bitwise left-shift
template <class T>
CUTE_HOST_DEVICE constexpr
auto
shiftl(T x, int s) {
  return s >= 0 ? (x << s) : (x >> -s);
}

// Computes the result of bitwise right-shift
template <class T>
CUTE_HOST_DEVICE constexpr
auto
shiftr(T x, int s) {
  return s >= 0 ? (x >> s) : (x << -s);
}

// Safe divide
// @pre t % u == 0
// @result t / u
template <class T, class U,
          __CUTE_REQUIRES(is_std_integral<T>::value &&
                          is_std_integral<U>::value)>
CUTE_HOST_DEVICE constexpr
auto
safe_div(T const& t, U const& u) {
  //assert(t % u == 0);
  return t / u;
}

/**
 * log2 computation
 */

template <class T>
CUTE_HOST_DEVICE constexpr
int32_t
log_2(T x) {
  assert(x > 0);
  static_assert(is_unsigned<T>::value, "Only to be used for unsigned integral types.");
  return static_cast<int32_t>(bit_width(x)) - 1;
}

template <class IntDiv, class IntMod>
struct DivModReturnType {
  IntDiv div_;
  IntMod mod_;
  CUTE_HOST_DEVICE constexpr
  DivModReturnType(IntDiv const& div, IntMod const& mod) : div_(div), mod_(mod) {}
};

// General divmod
template <class CInt0, class CInt1>
CUTE_HOST_DEVICE constexpr
auto
divmod(CInt0 const& a, CInt1 const& b) {
  return DivModReturnType{a / b, a % b};
}

// Specialized function with fastDivmod input
template <class CInt>
CUTE_HOST_DEVICE constexpr
auto
#if !defined(__HIP_PLATFORM_AMD__)
divmod(CInt const& a, cutlass::FastDivmod const& b) {
#else
divmod(CInt const& a, FastDivmod const& b) {
#endif
#if !defined(__HIP_PLATFORM_AMD__)
  using val_div_type = typename cutlass::FastDivmod::value_div_type;
  using val_mod_type = typename cutlass::FastDivmod::value_mod_type;
#else
  using val_div_type = typename FastDivmod::value_div_type;
  using val_mod_type = typename FastDivmod::value_mod_type;
#endif
  val_div_type div = 0;
  val_mod_type mod = 0;
  b(div, mod, a);
  return DivModReturnType{div, mod};
}

} // namespace cute
