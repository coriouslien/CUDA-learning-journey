// main.cu
#include "warmup.h"
#include "check_cuda.h"
#include "slab_allocator.h"

int main()
{
    warmup();
    slab_memory_pool();
    return 0;
}
