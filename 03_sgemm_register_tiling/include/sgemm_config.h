#pragma once

struct SgemmCfg
{
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 8;
    static constexpr int TM = 8;
    static constexpr int TN = 8;
    static constexpr int NUM_THREADS = (BM / TM) * (BN / TN); // 256
};
