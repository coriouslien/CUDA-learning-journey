
// pinned_buffer.h

#pragma once

#include "check_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cuda_fp16.h>

template <typename T>
class PinnedBuffer
{
private:
    T *data_ptr_ = nullptr;
    size_t n_ = 0;

public:
    explicit PinnedBuffer(size_t n);
    ~PinnedBuffer() noexcept;

    // no explicit copy constructor defined.
    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &) = delete; // shallow copy

    T *data();

    const T *data() const;

    size_t size() const;

    void initHostData(float value);
};
