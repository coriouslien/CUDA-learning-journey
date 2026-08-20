#pragma once
#include "nvtx3/nvtx3.hpp"

// define two marcos:
// one that automatically grabs the function name.
// another one lets us specifiy a custom name for smaller
// blocks of code inside a funtion.
#define NVTX_CONCAT_IMPL(x, y) x##y
#define NVTX_CONCAT(x, y) NVTX_CONCAT_IMPL(x, y)

#define NVTX_PROFILE_FUNC()                                \
    nvtx3::scoped_range NVTX_CONCAT(nvtx_scope_, __LINE__) \
    {                                                      \
        nvtx3::message { __func__ }                        \
    }

#define NVTX_PROFILE_SCOPE(name)                           \
    nvtx3::scoped_range NVTX_CONCAT(nvtx_scope_, __LINE__) \
    {                                                      \
        nvtx3::message { name }                            \
    }
