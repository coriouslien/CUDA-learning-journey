
// pinned_buffer.cu

#include "pinned_buffer.h"

#include <omp.h>

template <typename T>
PinnedBuffer<T>::PinnedBuffer(size_t n)
    : n_(n),
      data_ptr_(nullptr)

{
    if (n_ > 0)
    {
        CHECK_CUDA(cudaMallocHost(&data_ptr_, n_ * sizeof(T)));
    }
}
template <typename T>
PinnedBuffer<T>::~PinnedBuffer() noexcept
{
    if (data_ptr_)
        cudaFreeHost(data_ptr_);
}
template <typename T>
T *PinnedBuffer<T>::data()
{
    return data_ptr_;
}
template <typename T>
const T *PinnedBuffer<T>::data() const
{
    return data_ptr_;
}
template <typename T>
size_t PinnedBuffer<T>::size() const
{
    return n_;
}
template <typename T>
void PinnedBuffer<T>::initHostData(float value)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_; i++)
    {
        data_ptr_[i] = static_cast<T>(value);
    }
}

template class PinnedBuffer<__half>;
template class PinnedBuffer<double>;
template class PinnedBuffer<float>;
template class PinnedBuffer<int>;
