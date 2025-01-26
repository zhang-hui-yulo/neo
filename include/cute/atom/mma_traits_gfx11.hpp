#pragma once

#include <cute/arch/mma_gfx11.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

namespace {

// (T32,V16) -> (M16,N16)
using GFX11_16x16_Row = Layout<Shape <Shape <_16,_2>,Shape < _16>>,
                               Stride<Stride< _1,_0>,Stride< _16>>>;

}

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = fp16 * fp16 + fp16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN>
{
    using ValTypeD = half_t;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = half_t;

    struct CUTE_ALIGNAS(4) FrgTypeAccum {
        half_t value;
        half_t empty;

        CUTE_HOST_DEVICE constexpr operator half_t() const noexcept { return value; }
    };

    using FrgTypeC = FrgTypeAccum;

    using Shape_MNK = Shape<_16, _16, _16>;
    using ThrID = Layout<_32>;
    using ALayout = GFX11_16x16_Row;
    using BLayout = GFX11_16x16_Row;
    using CLayout = Layout<Shape <Shape <_16, _2>, Shape < _8>>,
                           Stride<Stride<_16, _1>, Stride< _2>>>;
};

}