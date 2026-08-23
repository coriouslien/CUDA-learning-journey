// main.cu

#include "warmup.h"
#include "wmma_double_buff_async.h"

int main()
{
    warmup();
    // int m = 256;
    // int n = 256;
    // int k = 256;
    // int m = 1024;
    // int n = 1024;
    // int k = 1024;
    // int m = 4096;
    // int n = 4096;
    // int k = 4096;
    // Performance: ~88.5 TFLOPS
    int m = 8192;
    int n = 8192;
    int k = 8192;
    // int m = 16384;
    // int n = 16384;
    // int k = 16384;
    const int ITERS = 100;

    WMMAMultiple wmmaMultiple(m, n, k);
    wmmaMultiple.initData();
    wmmaMultiple.hostToDevice();
    wmmaMultiple.runWMMABasic(ITERS);
//    wmmaMultiple.verifyResult();

    return 0;
}
