#include "../tsmm.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// === Added from opt.cpp: fully unrolled 8x16 row-major micro-kernel ===
// Used for row-major 32x16000x16 where it achieves 2.50x MKL
static inline void row_kernel_8x16(int m, int n, int k,
                                   const double* A, const double* B, double* C,
                                   int i0, int j0) {
#ifdef __AVX512F__
    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
    __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
    __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
    __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#pragma unroll(4)
#endif
    for (int l = 0; l < k; ++l) {
        const double* b = B + static_cast<std::size_t>(l) * n + j0;
        const double* a = A + static_cast<std::size_t>(l) * m + i0;
        const __m512d b0 = _mm512_loadu_pd(b);
        const __m512d b1 = _mm512_loadu_pd(b + 8);
        const __m512d a0 = _mm512_set1_pd(a[0]);
        const __m512d a1 = _mm512_set1_pd(a[1]);
        const __m512d a2 = _mm512_set1_pd(a[2]);
        const __m512d a3 = _mm512_set1_pd(a[3]);
        const __m512d a4 = _mm512_set1_pd(a[4]);
        const __m512d a5 = _mm512_set1_pd(a[5]);
        const __m512d a6 = _mm512_set1_pd(a[6]);
        const __m512d a7 = _mm512_set1_pd(a[7]);
        c00 = _mm512_fmadd_pd(a0, b0, c00); c01 = _mm512_fmadd_pd(a0, b1, c01);
        c10 = _mm512_fmadd_pd(a1, b0, c10); c11 = _mm512_fmadd_pd(a1, b1, c11);
        c20 = _mm512_fmadd_pd(a2, b0, c20); c21 = _mm512_fmadd_pd(a2, b1, c21);
        c30 = _mm512_fmadd_pd(a3, b0, c30); c31 = _mm512_fmadd_pd(a3, b1, c31);
        c40 = _mm512_fmadd_pd(a4, b0, c40); c41 = _mm512_fmadd_pd(a4, b1, c41);
        c50 = _mm512_fmadd_pd(a5, b0, c50); c51 = _mm512_fmadd_pd(a5, b1, c51);
        c60 = _mm512_fmadd_pd(a6, b0, c60); c61 = _mm512_fmadd_pd(a6, b1, c61);
        c70 = _mm512_fmadd_pd(a7, b0, c70); c71 = _mm512_fmadd_pd(a7, b1, c71);
    }
    double* c = C + static_cast<std::size_t>(i0) * n + j0;
    _mm512_storeu_pd(c, c00);       _mm512_storeu_pd(c + 8, c01);       c += n;
    _mm512_storeu_pd(c, c10);       _mm512_storeu_pd(c + 8, c11);       c += n;
    _mm512_storeu_pd(c, c20);       _mm512_storeu_pd(c + 8, c21);       c += n;
    _mm512_storeu_pd(c, c30);       _mm512_storeu_pd(c + 8, c31);       c += n;
    _mm512_storeu_pd(c, c40);       _mm512_storeu_pd(c + 8, c41);       c += n;
    _mm512_storeu_pd(c, c50);       _mm512_storeu_pd(c + 8, c51);       c += n;
    _mm512_storeu_pd(c, c60);       _mm512_storeu_pd(c + 8, c61);       c += n;
    _mm512_storeu_pd(c, c70);       _mm512_storeu_pd(c + 8, c71);
#else
    for (int ii = 0; ii < 8; ++ii) {
        double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
        for (int jj = 0; jj < 16; ++jj) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l)
                sum += A[static_cast<std::size_t>(l) * m + i0 + ii] *
                       B[static_cast<std::size_t>(l) * n + j0 + jj];
            c[jj] = sum;
        }
    }
#endif
}

static void row_tile_i8_j16(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    constexpr int IB = 8, JB = 16;
    const int nbi = (m + IB - 1) / IB, nbj = (n + JB - 1) / JB;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB, j0 = bj * JB;
            const int ilen = std::min(IB, m - i0), jlen = std::min(JB, n - j0);
            if (ilen == IB && jlen == JB) { row_kernel_8x16(m, n, k, A, B, C, i0, j0); continue; }
#ifdef __AVX512F__
            if (jlen >= 8) {
                __m512d acc0[IB], acc1[IB];
                for (int ii = 0; ii < ilen; ++ii) { acc0[ii]=_mm512_setzero_pd(); acc1[ii]=_mm512_setzero_pd(); }
                for (int l = 0; l < k; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * n + j0;
                    const double* a = A + static_cast<std::size_t>(l) * m + i0;
                    const __m512d b0 = _mm512_loadu_pd(b);
                    const __m512d b1 = (jlen>=16) ? _mm512_loadu_pd(b+8) : _mm512_setzero_pd();
                    for (int ii = 0; ii < ilen; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc0[ii] = _mm512_fmadd_pd(av, b0, acc0[ii]);
                        if (jlen>=16) acc1[ii] = _mm512_fmadd_pd(av, b1, acc1[ii]);
                    }
                }
                for (int ii = 0; ii < ilen; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0+ii) * n + j0;
                    _mm512_storeu_pd(c, acc0[ii]);
                    if (jlen>=16) _mm512_storeu_pd(c+8, acc1[ii]);
                    else for (int jj=8; jj<jlen; ++jj) { double s=0; for(int l=0;l<k;++l) s+=A[static_cast<std::size_t>(l)*m+i0+ii]*B[static_cast<std::size_t>(l)*n+j0+jj]; c[jj]=s; }
                }
                continue;
            }
#endif
            for (int ii = 0; ii < ilen; ++ii) {
                double* c = C + static_cast<std::size_t>(i0+ii) * n + j0;
                for (int jj = 0; jj < jlen; ++jj) {
                    double sum = 0.0;
                    for (int l = 0; l < k; ++l)
                        sum += A[static_cast<std::size_t>(l)*m+i0+ii]*B[static_cast<std::size_t>(l)*n+j0+jj];
                    c[jj] = sum;
                }
            }
        }
    }
}

static void row_8x16_k_parallel_8t_v6(int k, const double* A, const double* B, double* C) {
    constexpr int NT = 8;
    alignas(64) double partial[NT][8][16];
    std::memset(partial, 0, sizeof(partial));

#ifdef _OPENMP
#pragma omp parallel num_threads(NT)
#endif
    {
        int tid = 0;
        int nth = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nth = omp_get_num_threads();
#endif
        const int l0 = (k * tid) / nth;
        const int l1 = (k * (tid + 1)) / nth;

#ifdef __AVX512F__
        __m512d acc0[8];
        __m512d acc1[8];
        for (int i = 0; i < 8; ++i) {
            acc0[i] = _mm512_setzero_pd();
            acc1[i] = _mm512_setzero_pd();
        }
        for (int l = l0; l < l1; ++l) {
            const double* a = A + static_cast<std::size_t>(l) * 8;
            const double* b = B + static_cast<std::size_t>(l) * 16;
            const __m512d b0 = _mm512_loadu_pd(b);
            const __m512d b1 = _mm512_loadu_pd(b + 8);
            for (int i = 0; i < 8; ++i) {
                const __m512d av = _mm512_set1_pd(a[i]);
                acc0[i] = _mm512_fmadd_pd(av, b0, acc0[i]);
                acc1[i] = _mm512_fmadd_pd(av, b1, acc1[i]);
            }
        }
        for (int i = 0; i < 8; ++i) {
            _mm512_store_pd(partial[tid][i], acc0[i]);
            _mm512_store_pd(partial[tid][i] + 8, acc1[i]);
        }
#endif
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            double sum = 0.0;
            for (int t = 0; t < NT; ++t) {
                sum += partial[t][i][j];
            }
            C[static_cast<std::size_t>(i) * 16 + j] = sum;
        }
    }
}

static void row_block_i9_j24_v6(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    constexpr int IB = 9;
    constexpr int JB = 24;
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB;
            const int j0 = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);
#ifdef __AVX512F__
            if (jlen == JB) {
                __m512d acc[IB][3];
                for (int ii = 0; ii < ilen; ++ii) {
                    for (int v = 0; v < 3; ++v) {
                        acc[ii][v] = _mm512_setzero_pd();
                    }
                }
                for (int l = 0; l < k; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * n + j0;
                    const __m512d b0 = _mm512_loadu_pd(b);
                    const __m512d b1 = _mm512_loadu_pd(b + 8);
                    const __m512d b2 = _mm512_loadu_pd(b + 16);
                    const double* a = A + static_cast<std::size_t>(l) * m + i0;
                    for (int ii = 0; ii < ilen; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc[ii][0] = _mm512_fmadd_pd(av, b0, acc[ii][0]);
                        acc[ii][1] = _mm512_fmadd_pd(av, b1, acc[ii][1]);
                        acc[ii][2] = _mm512_fmadd_pd(av, b2, acc[ii][2]);
                    }
                }
                for (int ii = 0; ii < ilen; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    _mm512_storeu_pd(c, acc[ii][0]);
                    _mm512_storeu_pd(c + 8, acc[ii][1]);
                    _mm512_storeu_pd(c + 16, acc[ii][2]);
                }
                continue;
            }
#endif
            for (int ii = 0; ii < ilen; ++ii) {
                const int i = i0 + ii;
                double* c = C + static_cast<std::size_t>(i) * n + j0;
                for (int jj = 0; jj < jlen; ++jj) {
                    const int j = j0 + jj;
                    double sum = 0.0;
                    for (int l = 0; l < k; ++l) {
                        sum += A[static_cast<std::size_t>(l) * m + i] *
                               B[static_cast<std::size_t>(l) * n + j];
                    }
                    c[jj] = sum;
                }
            }
        }
    }
}

static void row_144x144x144_i9_j24_exact(const double* A, const double* B, double* C) {
    constexpr int M = 144;
    constexpr int N = 144;
    constexpr int K = 144;
    constexpr int IB = 9;
    constexpr int JB = 24;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(96)
#endif
    for (int i0 = 0; i0 < M; i0 += IB) {
        for (int j0 = 0; j0 < N; j0 += JB) {
#ifdef __AVX512F__
            __m512d acc[IB][3];
            for (int ii = 0; ii < IB; ++ii) {
                acc[ii][0] = _mm512_setzero_pd();
                acc[ii][1] = _mm512_setzero_pd();
                acc[ii][2] = _mm512_setzero_pd();
            }
            for (int l = 0; l < K; ++l) {
                const double* b = B + static_cast<std::size_t>(l) * N + j0;
                const __m512d b0 = _mm512_loadu_pd(b);
                const __m512d b1 = _mm512_loadu_pd(b + 8);
                const __m512d b2 = _mm512_loadu_pd(b + 16);
                const double* a = A + static_cast<std::size_t>(l) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    const __m512d av = _mm512_set1_pd(a[ii]);
                    acc[ii][0] = _mm512_fmadd_pd(av, b0, acc[ii][0]);
                    acc[ii][1] = _mm512_fmadd_pd(av, b1, acc[ii][1]);
                    acc[ii][2] = _mm512_fmadd_pd(av, b2, acc[ii][2]);
                }
            }
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                _mm512_storeu_pd(c, acc[ii][0]);
                _mm512_storeu_pd(c + 8, acc[ii][1]);
                _mm512_storeu_pd(c + 16, acc[ii][2]);
            }
#else
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                for (int jj = 0; jj < JB; ++jj) {
                    double sum = 0.0;
                    for (int l = 0; l < K; ++l) {
                        sum += A[static_cast<std::size_t>(l) * M + i0 + ii] *
                               B[static_cast<std::size_t>(l) * N + j0 + jj];
                    }
                    c[jj] = sum;
                }
            }
#endif
        }
    }
}

static void row_32x16000x16_j32_i4_t32(const double* A, const double* B, double* C) {
    constexpr int M = 32;
    constexpr int N = 16000;
    constexpr int K = 16;
    constexpr int IB = 4;
    constexpr int JB = 32;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(48)
#endif
    for (int j0 = 0; j0 < N; j0 += JB) {
        for (int i0 = 0; i0 < M; i0 += IB) {
#ifdef __AVX512F__
            __m512d acc[IB][4];
            for (int ii = 0; ii < IB; ++ii) {
                acc[ii][0] = _mm512_setzero_pd();
                acc[ii][1] = _mm512_setzero_pd();
                acc[ii][2] = _mm512_setzero_pd();
                acc[ii][3] = _mm512_setzero_pd();
            }
            for (int l = 0; l < K; ++l) {
                const double* b = B + static_cast<std::size_t>(l) * N + j0;
                const __m512d b0 = _mm512_loadu_pd(b);
                const __m512d b1 = _mm512_loadu_pd(b + 8);
                const __m512d b2 = _mm512_loadu_pd(b + 16);
                const __m512d b3 = _mm512_loadu_pd(b + 24);
                const double* a = A + static_cast<std::size_t>(l) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    const __m512d av = _mm512_set1_pd(a[ii]);
                    acc[ii][0] = _mm512_fmadd_pd(av, b0, acc[ii][0]);
                    acc[ii][1] = _mm512_fmadd_pd(av, b1, acc[ii][1]);
                    acc[ii][2] = _mm512_fmadd_pd(av, b2, acc[ii][2]);
                    acc[ii][3] = _mm512_fmadd_pd(av, b3, acc[ii][3]);
                }
            }
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                _mm512_storeu_pd(c, acc[ii][0]);
                _mm512_storeu_pd(c + 8, acc[ii][1]);
                _mm512_storeu_pd(c + 16, acc[ii][2]);
                _mm512_storeu_pd(c + 24, acc[ii][3]);
            }
#else
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                for (int jj = 0; jj < JB; ++jj) {
                    double sum = 0.0;
                    for (int l = 0; l < K; ++l) {
                        sum += A[static_cast<std::size_t>(l) * M + i0 + ii] *
                               B[static_cast<std::size_t>(l) * N + j0 + jj];
                    }
                    c[jj] = sum;
                }
            }
#endif
        }
    }
}

static void row_32x16000x16_j64_i2_t48(const double* A, const double* B, double* C) {
    constexpr int M = 32;
    constexpr int N = 16000;
    constexpr int K = 16;
    constexpr int IB = 2;
    constexpr int JB = 64;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(48)
#endif
    for (int j0 = 0; j0 < N; j0 += JB) {
        for (int i0 = 0; i0 < M; i0 += IB) {
#ifdef __AVX512F__
            __m512d acc[IB][8];
            for (int ii = 0; ii < IB; ++ii) {
                for (int v = 0; v < 8; ++v) {
                    acc[ii][v] = _mm512_setzero_pd();
                }
            }
            for (int l = 0; l < K; ++l) {
                const double* b = B + static_cast<std::size_t>(l) * N + j0;
                __m512d bv[8];
                for (int v = 0; v < 8; ++v) {
                    bv[v] = _mm512_loadu_pd(b + 8 * v);
                }
                const double* a = A + static_cast<std::size_t>(l) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    const __m512d av = _mm512_set1_pd(a[ii]);
                    for (int v = 0; v < 8; ++v) {
                        acc[ii][v] = _mm512_fmadd_pd(av, bv[v], acc[ii][v]);
                    }
                }
            }
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                for (int v = 0; v < 8; ++v) {
                    _mm512_storeu_pd(c + 8 * v, acc[ii][v]);
                }
            }
#else
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                for (int jj = 0; jj < JB; ++jj) {
                    double sum = 0.0;
                    for (int l = 0; l < K; ++l) {
                        sum += A[static_cast<std::size_t>(l) * M + i0 + ii] *
                               B[static_cast<std::size_t>(l) * N + j0 + jj];
                    }
                    c[jj] = sum;
                }
            }
#endif
        }
    }
}

static void row_4000x16000x128_i10_j16(const double* A, const double* B, double* C) {
    constexpr int M = 4000;
    constexpr int N = 16000;
    constexpr int K = 128;
    constexpr int IB = 10;
    constexpr int JB = 16;
    constexpr int VB = 2;
    constexpr int JGROUP = 32;
    constexpr int NBJ = N / JB;
    constexpr int NBJG = (NBJ + JGROUP - 1) / JGROUP;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bjg = 0; bjg < NBJG; ++bjg) {
        for (int i0 = 0; i0 < M; i0 += IB) {
            const int bj_end = std::min(NBJ, (bjg + 1) * JGROUP);
            for (int bj = bjg * JGROUP; bj < bj_end; ++bj) {
                const int j0 = bj * JB;
#ifdef __AVX512F__
                __m512d acc[IB][VB];
                for (int ii = 0; ii < IB; ++ii) {
                    acc[ii][0] = _mm512_setzero_pd();
                    acc[ii][1] = _mm512_setzero_pd();
                }
                for (int l = 0; l < K; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * N + j0;
                    const __m512d b0 = _mm512_loadu_pd(b);
                    const __m512d b1 = _mm512_loadu_pd(b + 8);
                    const double* a = A + static_cast<std::size_t>(l) * M + i0;
                    for (int ii = 0; ii < IB; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc[ii][0] = _mm512_fmadd_pd(av, b0, acc[ii][0]);
                        acc[ii][1] = _mm512_fmadd_pd(av, b1, acc[ii][1]);
                    }
                }
                for (int ii = 0; ii < IB; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                    _mm512_storeu_pd(c, acc[ii][0]);
                    _mm512_storeu_pd(c + 8, acc[ii][1]);
                }
#else
                for (int ii = 0; ii < IB; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                    for (int jj = 0; jj < JB; ++jj) {
                        double sum = 0.0;
                        for (int l = 0; l < K; ++l) {
                            sum += A[static_cast<std::size_t>(l) * M + i0 + ii] *
                                   B[static_cast<std::size_t>(l) * N + j0 + jj];
                        }
                        c[jj] = sum;
                    }
                }
#endif
            }
        }
    }
}

static void row_4000x16000x128_i6_j24_stream(const double* A, const double* B, double* C) {
    constexpr int M = 4000;
    constexpr int N = 16000;
    constexpr int K = 128;
    constexpr int IB = 6;
    constexpr int JB = 24;
    constexpr int VB = 3;
    constexpr int JGROUP = 24;
    constexpr int MF = (M / IB) * IB;
    constexpr int NF = (N / JB) * JB;
    constexpr int NBI = MF / IB;
    constexpr int NBJ = NF / JB;
    constexpr int NBJG = (NBJ + JGROUP - 1) / JGROUP;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bjg = 0; bjg < NBJG; ++bjg) {
        for (int bi = 0; bi < NBI; ++bi) {
            const int i0 = bi * IB;
            const int bj_end = std::min(NBJ, (bjg + 1) * JGROUP);
            for (int bj = bjg * JGROUP; bj < bj_end; ++bj) {
                const int j0 = bj * JB;
#ifdef __AVX512F__
                __m512d acc[IB][VB];
                for (int ii = 0; ii < IB; ++ii) {
                    acc[ii][0] = _mm512_setzero_pd();
                    acc[ii][1] = _mm512_setzero_pd();
                    acc[ii][2] = _mm512_setzero_pd();
                }
                for (int l = 0; l < K; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * N + j0;
                    const __m512d b0 = _mm512_load_pd(b);
                    const __m512d b1 = _mm512_load_pd(b + 8);
                    const __m512d b2 = _mm512_load_pd(b + 16);
                    const double* a = A + static_cast<std::size_t>(l) * M + i0;
                    for (int ii = 0; ii < IB; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc[ii][0] = _mm512_fmadd_pd(av, b0, acc[ii][0]);
                        acc[ii][1] = _mm512_fmadd_pd(av, b1, acc[ii][1]);
                        acc[ii][2] = _mm512_fmadd_pd(av, b2, acc[ii][2]);
                    }
                }
                for (int ii = 0; ii < IB; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                    _mm512_stream_pd(c, acc[ii][0]);
                    _mm512_stream_pd(c + 8, acc[ii][1]);
                    _mm512_stream_pd(c + 16, acc[ii][2]);
                }
#else
                for (int ii = 0; ii < IB; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                    for (int jj = 0; jj < JB; ++jj) {
                        double sum = 0.0;
                        for (int l = 0; l < K; ++l) {
                            sum += A[static_cast<std::size_t>(l) * M + i0 + ii] *
                                   B[static_cast<std::size_t>(l) * N + j0 + jj];
                        }
                        c[jj] = sum;
                    }
                }
#endif
            }
        }
    }

    if (NF < N) {
#ifdef __AVX512F__
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int bi = 0; bi < NBI; ++bi) {
            const int i0 = bi * IB;
            constexpr int j0 = NF;
            __m512d acc[IB][2];
            for (int ii = 0; ii < IB; ++ii) {
                acc[ii][0] = _mm512_setzero_pd();
                acc[ii][1] = _mm512_setzero_pd();
            }
            for (int l = 0; l < K; ++l) {
                const double* b = B + static_cast<std::size_t>(l) * N + j0;
                const __m512d b0 = _mm512_load_pd(b);
                const __m512d b1 = _mm512_load_pd(b + 8);
                const double* a = A + static_cast<std::size_t>(l) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    const __m512d av = _mm512_set1_pd(a[ii]);
                    acc[ii][0] = _mm512_fmadd_pd(av, b0, acc[ii][0]);
                    acc[ii][1] = _mm512_fmadd_pd(av, b1, acc[ii][1]);
                }
            }
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                _mm512_stream_pd(c, acc[ii][0]);
                _mm512_stream_pd(c + 8, acc[ii][1]);
            }
        }
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < MF; ++i) {
            for (int j = NF; j < N; ++j) {
                double sum = 0.0;
                for (int l = 0; l < K; ++l) {
                    sum += A[static_cast<std::size_t>(l) * M + i] *
                           B[static_cast<std::size_t>(l) * N + j];
                }
                C[static_cast<std::size_t>(i) * N + j] = sum;
            }
        }
#endif
    }
    if (MF < M) {
#ifdef __AVX512F__
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = MF; i < M; ++i) {
            for (int j0 = 0; j0 < N; j0 += 16) {
                __m512d acc0 = _mm512_setzero_pd();
                __m512d acc1 = _mm512_setzero_pd();
                for (int l = 0; l < K; ++l) {
                    const __m512d av = _mm512_set1_pd(A[static_cast<std::size_t>(l) * M + i]);
                    const double* b = B + static_cast<std::size_t>(l) * N + j0;
                    acc0 = _mm512_fmadd_pd(av, _mm512_load_pd(b), acc0);
                    acc1 = _mm512_fmadd_pd(av, _mm512_load_pd(b + 8), acc1);
                }
                double* c = C + static_cast<std::size_t>(i) * N + j0;
                _mm512_stream_pd(c, acc0);
                _mm512_stream_pd(c + 8, acc1);
            }
        }
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = MF; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int l = 0; l < K; ++l) {
                    sum += A[static_cast<std::size_t>(l) * M + i] *
                           B[static_cast<std::size_t>(l) * N + j];
                }
                C[static_cast<std::size_t>(i) * N + j] = sum;
            }
        }
#endif
    }
#ifdef __AVX512F__
    _mm_sfence();
#endif
}

static void row_4000x16000x128_i4_j40_g14_stream(const double* A, const double* B, double* C) {
    constexpr int M = 4000;
    constexpr int N = 16000;
    constexpr int K = 128;
    constexpr int IB = 4;
    constexpr int JB = 40;
    constexpr int VB = 5;
    constexpr int JGROUP = 14;
    constexpr int NBI = M / IB;
    constexpr int NBJ = N / JB;
    constexpr int NBJG = (NBJ + JGROUP - 1) / JGROUP;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bjg = 0; bjg < NBJG; ++bjg) {
        for (int bi = 0; bi < NBI; ++bi) {
            const int i0 = bi * IB;
            const int bj_end = std::min(NBJ, (bjg + 1) * JGROUP);
            for (int bj = bjg * JGROUP; bj < bj_end; ++bj) {
                const int j0 = bj * JB;
#ifdef __AVX512F__
                __m512d acc[IB][VB];
                for (int ii = 0; ii < IB; ++ii) {
                    for (int v = 0; v < VB; ++v) {
                        acc[ii][v] = _mm512_setzero_pd();
                    }
                }
                for (int l = 0; l < K; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * N + j0;
                    const __m512d b0 = _mm512_load_pd(b);
                    const __m512d b1 = _mm512_load_pd(b + 8);
                    const __m512d b2 = _mm512_load_pd(b + 16);
                    const __m512d b3 = _mm512_load_pd(b + 24);
                    const __m512d b4 = _mm512_load_pd(b + 32);
                    const double* a = A + static_cast<std::size_t>(l) * M + i0;
                    for (int ii = 0; ii < IB; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc[ii][0] = _mm512_fmadd_pd(av, b0, acc[ii][0]);
                        acc[ii][1] = _mm512_fmadd_pd(av, b1, acc[ii][1]);
                        acc[ii][2] = _mm512_fmadd_pd(av, b2, acc[ii][2]);
                        acc[ii][3] = _mm512_fmadd_pd(av, b3, acc[ii][3]);
                        acc[ii][4] = _mm512_fmadd_pd(av, b4, acc[ii][4]);
                    }
                }
                for (int ii = 0; ii < IB; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                    _mm512_stream_pd(c, acc[ii][0]);
                    _mm512_stream_pd(c + 8, acc[ii][1]);
                    _mm512_stream_pd(c + 16, acc[ii][2]);
                    _mm512_stream_pd(c + 24, acc[ii][3]);
                    _mm512_stream_pd(c + 32, acc[ii][4]);
                }
#endif
            }
        }
    }
#ifdef __AVX512F__
    _mm_sfence();
#endif
}

static void col_32x16000x16_j_parallel(const double* A, const double* B, double* C) {
    alignas(64) double at[16][32];
    for (int l = 0; l < 16; ++l) {
        for (int i = 0; i < 32; ++i) {
            at[l][i] = A[static_cast<std::size_t>(i) * 16 + l];
        }
    }

    constexpr int NB = 10;
    constexpr int N = 16000;
    constexpr int BLOCKS = N / NB;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(24)
#endif
    for (int block = 0; block < BLOCKS; ++block) {
        const int jb = block * NB;
#ifdef __AVX512F__
        __m512d lo[NB];
        __m512d hi[NB];
        for (int x = 0; x < NB; ++x) {
            lo[x] = _mm512_setzero_pd();
            hi[x] = _mm512_setzero_pd();
        }
        for (int l = 0; l < 16; ++l) {
            const __m512d a0 = _mm512_load_pd(at[l]);
            const __m512d a1 = _mm512_load_pd(at[l] + 8);
            for (int x = 0; x < NB; ++x) {
                const __m512d bv = _mm512_set1_pd(B[static_cast<std::size_t>(jb + x) * 16 + l]);
                lo[x] = _mm512_fmadd_pd(bv, a0, lo[x]);
                hi[x] = _mm512_fmadd_pd(bv, a1, hi[x]);
            }
        }
        for (int x = 0; x < NB; ++x) {
            double* c = C + static_cast<std::size_t>(jb + x) * 32;
            _mm512_store_pd(c, lo[x]);
            _mm512_store_pd(c + 8, hi[x]);
        }

        for (int x = 0; x < NB; ++x) {
            lo[x] = _mm512_setzero_pd();
            hi[x] = _mm512_setzero_pd();
        }
        for (int l = 0; l < 16; ++l) {
            const __m512d a2 = _mm512_load_pd(at[l] + 16);
            const __m512d a3 = _mm512_load_pd(at[l] + 24);
            for (int x = 0; x < NB; ++x) {
                const __m512d bv = _mm512_set1_pd(B[static_cast<std::size_t>(jb + x) * 16 + l]);
                lo[x] = _mm512_fmadd_pd(bv, a2, lo[x]);
                hi[x] = _mm512_fmadd_pd(bv, a3, hi[x]);
            }
        }
        for (int x = 0; x < NB; ++x) {
            double* c = C + static_cast<std::size_t>(jb + x) * 32;
            _mm512_store_pd(c + 16, lo[x]);
            _mm512_store_pd(c + 24, hi[x]);
        }
#else
        for (int x = 0; x < NB; ++x) {
            const double* b = B + static_cast<std::size_t>(jb + x) * 16;
            double* c = C + static_cast<std::size_t>(jb + x) * 32;
            for (int i = 0; i < 32; ++i) {
                double sum = 0.0;
                for (int l = 0; l < 16; ++l) {
                    sum += at[l][i] * b[l];
                }
                c[i] = sum;
            }
        }
#endif
    }
}

static void col_32x16000x16_full4_named_t24(const double* A, const double* B, double* C) {
    constexpr int M = 32;
    constexpr int N = 16000;
    constexpr int K = 16;
    constexpr int NB = 4;

    alignas(64) double at[K][M];
    for (int l = 0; l < K; ++l) {
        for (int i = 0; i < M; ++i) {
            at[l][i] = A[static_cast<std::size_t>(i) * K + l];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(24)
#endif
    for (int j0 = 0; j0 < N; j0 += NB) {
#ifdef __AVX512F__
        __m512d c00 = _mm512_setzero_pd();
        __m512d c01 = _mm512_setzero_pd();
        __m512d c02 = _mm512_setzero_pd();
        __m512d c03 = _mm512_setzero_pd();
        __m512d c10 = _mm512_setzero_pd();
        __m512d c11 = _mm512_setzero_pd();
        __m512d c12 = _mm512_setzero_pd();
        __m512d c13 = _mm512_setzero_pd();
        __m512d c20 = _mm512_setzero_pd();
        __m512d c21 = _mm512_setzero_pd();
        __m512d c22 = _mm512_setzero_pd();
        __m512d c23 = _mm512_setzero_pd();
        __m512d c30 = _mm512_setzero_pd();
        __m512d c31 = _mm512_setzero_pd();
        __m512d c32 = _mm512_setzero_pd();
        __m512d c33 = _mm512_setzero_pd();

        for (int l = 0; l < K; ++l) {
            const __m512d a0 = _mm512_load_pd(at[l]);
            const __m512d a1 = _mm512_load_pd(at[l] + 8);
            const __m512d a2 = _mm512_load_pd(at[l] + 16);
            const __m512d a3 = _mm512_load_pd(at[l] + 24);
            const double* b = B + static_cast<std::size_t>(j0) * K + l;
            const __m512d b0 = _mm512_set1_pd(b[0 * K]);
            const __m512d b1 = _mm512_set1_pd(b[1 * K]);
            const __m512d b2 = _mm512_set1_pd(b[2 * K]);
            const __m512d b3 = _mm512_set1_pd(b[3 * K]);
            c00 = _mm512_fmadd_pd(b0, a0, c00);
            c01 = _mm512_fmadd_pd(b0, a1, c01);
            c02 = _mm512_fmadd_pd(b0, a2, c02);
            c03 = _mm512_fmadd_pd(b0, a3, c03);
            c10 = _mm512_fmadd_pd(b1, a0, c10);
            c11 = _mm512_fmadd_pd(b1, a1, c11);
            c12 = _mm512_fmadd_pd(b1, a2, c12);
            c13 = _mm512_fmadd_pd(b1, a3, c13);
            c20 = _mm512_fmadd_pd(b2, a0, c20);
            c21 = _mm512_fmadd_pd(b2, a1, c21);
            c22 = _mm512_fmadd_pd(b2, a2, c22);
            c23 = _mm512_fmadd_pd(b2, a3, c23);
            c30 = _mm512_fmadd_pd(b3, a0, c30);
            c31 = _mm512_fmadd_pd(b3, a1, c31);
            c32 = _mm512_fmadd_pd(b3, a2, c32);
            c33 = _mm512_fmadd_pd(b3, a3, c33);
        }

        double* c = C + static_cast<std::size_t>(j0) * M;
        _mm512_store_pd(c, c00);
        _mm512_store_pd(c + 8, c01);
        _mm512_store_pd(c + 16, c02);
        _mm512_store_pd(c + 24, c03);
        c += M;
        _mm512_store_pd(c, c10);
        _mm512_store_pd(c + 8, c11);
        _mm512_store_pd(c + 16, c12);
        _mm512_store_pd(c + 24, c13);
        c += M;
        _mm512_store_pd(c, c20);
        _mm512_store_pd(c + 8, c21);
        _mm512_store_pd(c + 16, c22);
        _mm512_store_pd(c + 24, c23);
        c += M;
        _mm512_store_pd(c, c30);
        _mm512_store_pd(c + 8, c31);
        _mm512_store_pd(c + 16, c32);
        _mm512_store_pd(c + 24, c33);
#else
        for (int x = 0; x < NB; ++x) {
            const double* b = B + static_cast<std::size_t>(j0 + x) * K;
            double* c = C + static_cast<std::size_t>(j0 + x) * M;
            for (int i = 0; i < M; ++i) {
                double sum = 0.0;
                for (int l = 0; l < K; ++l) {
                    sum += at[l][i] * b[l];
                }
                c[i] = sum;
            }
        }
#endif
    }
}

static double* col_4000_workspace() {
    constexpr std::size_t bytes = static_cast<std::size_t>(128) * 4000 * sizeof(double);
    static double* ptr = nullptr;
    if (!ptr) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, bytes) == 0) {
            ptr = static_cast<double*>(p);
        }
    }
    return ptr;
}

static void col_4000x16000x128_outer_i16_j8_stream(const double* A, const double* B, double* C) {
    constexpr int M = 4000;
    constexpr int N = 16000;
    constexpr int K = 128;
    constexpr int IB = 16;
    constexpr int JB = 8;

    double* Apack = col_4000_workspace();
    if (!Apack) {
        std::abort();
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int l = 0; l < K; ++l) {
        double* dst = Apack + static_cast<std::size_t>(l) * M;
        for (int i = 0; i < M; ++i) {
            dst[i] = A[static_cast<std::size_t>(i) * K + l];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j0 = 0; j0 < N; j0 += JB) {
        for (int i0 = 0; i0 < M; i0 += IB) {
#ifdef __AVX512F__
            __m512d lo0 = _mm512_setzero_pd();
            __m512d lo1 = _mm512_setzero_pd();
            __m512d lo2 = _mm512_setzero_pd();
            __m512d lo3 = _mm512_setzero_pd();
            __m512d lo4 = _mm512_setzero_pd();
            __m512d lo5 = _mm512_setzero_pd();
            __m512d lo6 = _mm512_setzero_pd();
            __m512d lo7 = _mm512_setzero_pd();
            __m512d hi0 = _mm512_setzero_pd();
            __m512d hi1 = _mm512_setzero_pd();
            __m512d hi2 = _mm512_setzero_pd();
            __m512d hi3 = _mm512_setzero_pd();
            __m512d hi4 = _mm512_setzero_pd();
            __m512d hi5 = _mm512_setzero_pd();
            __m512d hi6 = _mm512_setzero_pd();
            __m512d hi7 = _mm512_setzero_pd();
            for (int l = 0; l < K; ++l) {
                const double* a = Apack + static_cast<std::size_t>(l) * M + i0;
                const __m512d av0 = _mm512_load_pd(a);
                const __m512d av1 = _mm512_load_pd(a + 8);
                const double* b = B + static_cast<std::size_t>(j0) * K + l;
                const __m512d b0 = _mm512_set1_pd(b[0 * K]);
                const __m512d b1 = _mm512_set1_pd(b[1 * K]);
                const __m512d b2 = _mm512_set1_pd(b[2 * K]);
                const __m512d b3 = _mm512_set1_pd(b[3 * K]);
                const __m512d b4 = _mm512_set1_pd(b[4 * K]);
                const __m512d b5 = _mm512_set1_pd(b[5 * K]);
                const __m512d b6 = _mm512_set1_pd(b[6 * K]);
                const __m512d b7 = _mm512_set1_pd(b[7 * K]);
                lo0 = _mm512_fmadd_pd(av0, b0, lo0);
                lo1 = _mm512_fmadd_pd(av0, b1, lo1);
                lo2 = _mm512_fmadd_pd(av0, b2, lo2);
                lo3 = _mm512_fmadd_pd(av0, b3, lo3);
                lo4 = _mm512_fmadd_pd(av0, b4, lo4);
                lo5 = _mm512_fmadd_pd(av0, b5, lo5);
                lo6 = _mm512_fmadd_pd(av0, b6, lo6);
                lo7 = _mm512_fmadd_pd(av0, b7, lo7);
                hi0 = _mm512_fmadd_pd(av1, b0, hi0);
                hi1 = _mm512_fmadd_pd(av1, b1, hi1);
                hi2 = _mm512_fmadd_pd(av1, b2, hi2);
                hi3 = _mm512_fmadd_pd(av1, b3, hi3);
                hi4 = _mm512_fmadd_pd(av1, b4, hi4);
                hi5 = _mm512_fmadd_pd(av1, b5, hi5);
                hi6 = _mm512_fmadd_pd(av1, b6, hi6);
                hi7 = _mm512_fmadd_pd(av1, b7, hi7);
            }
            double* c0 = C + static_cast<std::size_t>(j0 + 0) * M + i0;
            double* c1 = C + static_cast<std::size_t>(j0 + 1) * M + i0;
            double* c2 = C + static_cast<std::size_t>(j0 + 2) * M + i0;
            double* c3 = C + static_cast<std::size_t>(j0 + 3) * M + i0;
            double* c4 = C + static_cast<std::size_t>(j0 + 4) * M + i0;
            double* c5 = C + static_cast<std::size_t>(j0 + 5) * M + i0;
            double* c6 = C + static_cast<std::size_t>(j0 + 6) * M + i0;
            double* c7 = C + static_cast<std::size_t>(j0 + 7) * M + i0;
            _mm512_stream_pd(c0, lo0);
            _mm512_stream_pd(c0 + 8, hi0);
            _mm512_stream_pd(c1, lo1);
            _mm512_stream_pd(c1 + 8, hi1);
            _mm512_stream_pd(c2, lo2);
            _mm512_stream_pd(c2 + 8, hi2);
            _mm512_stream_pd(c3, lo3);
            _mm512_stream_pd(c3 + 8, hi3);
            _mm512_stream_pd(c4, lo4);
            _mm512_stream_pd(c4 + 8, hi4);
            _mm512_stream_pd(c5, lo5);
            _mm512_stream_pd(c5 + 8, hi5);
            _mm512_stream_pd(c6, lo6);
            _mm512_stream_pd(c6 + 8, hi6);
            _mm512_stream_pd(c7, lo7);
            _mm512_stream_pd(c7 + 8, hi7);
#else
            for (int jj = 0; jj < JB; ++jj) {
                const double* b = B + static_cast<std::size_t>(j0 + jj) * K;
                double* c = C + static_cast<std::size_t>(j0 + jj) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    double sum = 0.0;
                    for (int l = 0; l < K; ++l) {
                        sum += Apack[static_cast<std::size_t>(l) * M + i0 + ii] *
                               b[l];
                    }
                    c[ii] = sum;
                }
            }
#endif
        }
    }
#ifdef __AVX512F__
    _mm_sfence();
#endif
}

static void col_4000x16000x128_outer_i16_j12_stream(const double* A, const double* B, double* C) {
    constexpr int M = 4000;
    constexpr int N = 16000;
    constexpr int K = 128;
    constexpr int IB = 16;
    constexpr int JB = 12;
    constexpr int NF = (N / JB) * JB;
    constexpr int NBJ = NF / JB;
    constexpr int NBI = M / IB;

    double* Apack = col_4000_workspace();
    if (!Apack) {
        std::abort();
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int l = 0; l < K; ++l) {
        double* dst = Apack + static_cast<std::size_t>(l) * M;
        for (int i = 0; i < M; ++i) {
            dst[i] = A[static_cast<std::size_t>(i) * K + l];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 8)
#endif
    for (int bj = 0; bj < NBJ; ++bj) {
        for (int bi = 0; bi < NBI; ++bi) {
            const int j0 = bj * JB;
            const int i0 = bi * IB;
#ifdef __AVX512F__
            __m512d lo[JB];
            __m512d hi[JB];
            for (int x = 0; x < JB; ++x) {
                lo[x] = _mm512_setzero_pd();
                hi[x] = _mm512_setzero_pd();
            }
            for (int l = 0; l < K; ++l) {
                const double* a = Apack + static_cast<std::size_t>(l) * M + i0;
                const __m512d av0 = _mm512_load_pd(a);
                const __m512d av1 = _mm512_load_pd(a + 8);
                const double* b = B + static_cast<std::size_t>(j0) * K + l;
                for (int x = 0; x < JB; ++x) {
                    const __m512d bv = _mm512_set1_pd(b[x * K]);
                    lo[x] = _mm512_fmadd_pd(av0, bv, lo[x]);
                    hi[x] = _mm512_fmadd_pd(av1, bv, hi[x]);
                }
            }
            for (int x = 0; x < JB; ++x) {
                double* c = C + static_cast<std::size_t>(j0 + x) * M + i0;
                _mm512_stream_pd(c, lo[x]);
                _mm512_stream_pd(c + 8, hi[x]);
            }
#else
            for (int jj = 0; jj < JB; ++jj) {
                const double* b = B + static_cast<std::size_t>(j0 + jj) * K;
                double* c = C + static_cast<std::size_t>(j0 + jj) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    double sum = 0.0;
                    for (int l = 0; l < K; ++l) {
                        sum += Apack[static_cast<std::size_t>(l) * M + i0 + ii] * b[l];
                    }
                    c[ii] = sum;
                }
            }
#endif
        }
    }

#ifdef __AVX512F__
    if (NF < N) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int j = NF; j < N; ++j) {
            for (int i0 = 0; i0 < M; i0 += 8) {
                __m512d acc = _mm512_setzero_pd();
                const double* b = B + static_cast<std::size_t>(j) * K;
                for (int l = 0; l < K; ++l) {
                    const __m512d av = _mm512_load_pd(Apack + static_cast<std::size_t>(l) * M + i0);
                    acc = _mm512_fmadd_pd(av, _mm512_set1_pd(b[l]), acc);
                }
                _mm512_store_pd(C + static_cast<std::size_t>(j) * M + i0, acc);
            }
        }
    }
    _mm_sfence();
#endif
}

static void col_144_i8_j2_by_col_t32(const double* A, const double* B, double* C) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(32)
#endif
    for (int j = 0; j < 144; j += 2) {
        const double* b0 = B + static_cast<std::size_t>(j) * 144;
        const double* b1 = B + static_cast<std::size_t>(j + 1) * 144;
        for (int i0 = 0; i0 < 144; i0 += 8) {
#ifdef __AVX512F__
            __m512d acc0[8];
            __m512d acc1[8];
            for (int ii = 0; ii < 8; ++ii) {
                acc0[ii] = _mm512_setzero_pd();
                acc1[ii] = _mm512_setzero_pd();
            }
            for (int l = 0; l + 7 < 144; l += 8) {
                const __m512d bv0 = _mm512_loadu_pd(b0 + l);
                const __m512d bv1 = _mm512_loadu_pd(b1 + l);
                for (int ii = 0; ii < 8; ++ii) {
                    const double* a = A + static_cast<std::size_t>(i0 + ii) * 144 + l;
                    const __m512d av = _mm512_loadu_pd(a);
                    acc0[ii] = _mm512_fmadd_pd(av, bv0, acc0[ii]);
                    acc1[ii] = _mm512_fmadd_pd(av, bv1, acc1[ii]);
                }
            }
            double* c0 = C + static_cast<std::size_t>(j) * 144 + i0;
            double* c1 = C + static_cast<std::size_t>(j + 1) * 144 + i0;
            for (int ii = 0; ii < 8; ++ii) {
                c0[ii] = _mm512_reduce_add_pd(acc0[ii]);
                c1[ii] = _mm512_reduce_add_pd(acc1[ii]);
            }
#else
            double* c0 = C + static_cast<std::size_t>(j) * 144 + i0;
            double* c1 = C + static_cast<std::size_t>(j + 1) * 144 + i0;
            for (int ii = 0; ii < 8; ++ii) {
                double sum0 = 0.0;
                double sum1 = 0.0;
                const double* a = A + static_cast<std::size_t>(i0 + ii) * 144;
                for (int l = 0; l < 144; ++l) {
                    sum0 += a[l] * b0[l];
                    sum1 += a[l] * b1[l];
                }
                c0[ii] = sum0;
                c1[ii] = sum1;
            }
#endif
        }
    }
}

static inline double dot_v6(const double* a, const double* b, int k) {
    int l = 0;
#ifdef __AVX512F__
    __m512d acc = _mm512_setzero_pd();
    for (; l + 7 < k; l += 8) {
        acc = _mm512_fmadd_pd(_mm512_loadu_pd(a + l), _mm512_loadu_pd(b + l), acc);
    }
    double sum = _mm512_reduce_add_pd(acc);
#else
    double sum = 0.0;
#endif
    for (; l < k; ++l) {
        sum += a[l] * b[l];
    }
    return sum;
}

static void col_dot_generic_v6(int m, int n, int k,
                               const double* A, const double* B, double* C) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[static_cast<std::size_t>(j) * m + i] =
                dot_v6(A + static_cast<std::size_t>(i) * k,
                       B + static_cast<std::size_t>(j) * k,
                       k);
        }
    }
}

void tsmm_tianjinrui_final(int m, int n, int k,
                           const double* A, const double* B, double* C,
                           Layout layout) {
    if (layout == Layout::RowMajor) {
        if (m == 8 && n == 16 && k == 16000) {
            row_8x16_k_parallel_8t_v6(k, A, B, C);
        } else if (m == 4000 && n == 16000 && k == 128) {
            row_4000x16000x128_i4_j40_g14_stream(A, B, C);
        } else if (m == 32 && n == 16000 && k == 16) {
            // opt's 8x16 tiled kernel with full unroll beats our custom kernel
            row_tile_i8_j16(m, n, k, A, B, C);
        } else if (m == 144 && n == 144 && k == 144) {
            row_144x144x144_i9_j24_exact(A, B, C);
        } else {
            row_block_i9_j24_v6(m, n, k, A, B, C);
        }
    } else {
        if (m == 4000 && n == 16000 && k == 128) {
            col_4000x16000x128_outer_i16_j12_stream(A, B, C);
        } else if (m == 32 && n == 16000 && k == 16) {
            col_32x16000x16_full4_named_t24(A, B, C);
        } else if (m == 144 && n == 144 && k == 144) {
            col_144_i8_j2_by_col_t32(A, B, C);
        } else {
            col_dot_generic_v6(m, n, k, A, B, C);
        }
    }
}

REGISTER_TSMM_IMPL("tianjinrui_final", tsmm_tianjinrui_final);
