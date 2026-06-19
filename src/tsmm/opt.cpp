#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static int limited_threads(int limit) {
#ifdef _OPENMP
    return std::max(1, std::min(omp_get_max_threads(), limit));
#else
    (void)limit;
    return 1;
#endif
}

static inline double hsum_vec(
#ifdef __AVX512F__
    __m512d v
#else
    double v
#endif
) {
#ifdef __AVX512F__
    return _mm512_reduce_add_pd(v);
#else
    return v;
#endif
}

static inline double dot_contiguous(const double* a, const double* b, int k) {
    int l = 0;
#ifdef __AVX512F__
    __m512d acc = _mm512_setzero_pd();
    for (; l + 7 < k; l += 8) {
        acc = _mm512_fmadd_pd(_mm512_loadu_pd(a + l), _mm512_loadu_pd(b + l), acc);
    }
    double sum = hsum_vec(acc);
#else
    double sum = 0.0;
#endif
    for (; l < k; ++l) {
        sum += a[l] * b[l];
    }
    return sum;
}

static inline void row_store_block_tail(int m, int n, int k,
                                        const double* A, const double* B, double* C,
                                        int i0, int j0, int ilen, int jlen) {
    for (int ii = 0; ii < ilen; ++ii) {
        double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
        for (int jj = 0; jj < jlen; ++jj) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += A[static_cast<std::size_t>(l) * m + i0 + ii] *
                       B[static_cast<std::size_t>(l) * n + j0 + jj];
            }
            c[jj] = sum;
        }
    }
}

static void row_tile_i8_j16(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    constexpr int IB = 8;
    constexpr int JB = 16;
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
            if (jlen >= 8) {
                __m512d acc0[IB];
                __m512d acc1[IB];
                for (int ii = 0; ii < ilen; ++ii) {
                    acc0[ii] = _mm512_setzero_pd();
                    acc1[ii] = _mm512_setzero_pd();
                }

                for (int l = 0; l < k; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * n + j0;
                    const double* a = A + static_cast<std::size_t>(l) * m + i0;
                    const __m512d b0 = _mm512_loadu_pd(b);
                    const __m512d b1 = (jlen >= 16) ? _mm512_loadu_pd(b + 8) : _mm512_setzero_pd();
                    for (int ii = 0; ii < ilen; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc0[ii] = _mm512_fmadd_pd(av, b0, acc0[ii]);
                        if (jlen >= 16) {
                            acc1[ii] = _mm512_fmadd_pd(av, b1, acc1[ii]);
                        }
                    }
                }

                for (int ii = 0; ii < ilen; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    _mm512_storeu_pd(c, acc0[ii]);
                    if (jlen >= 16) {
                        _mm512_storeu_pd(c + 8, acc1[ii]);
                    } else {
                        row_store_block_tail(m, n, k, A, B, C, i0 + ii, j0 + 8, 1, jlen - 8);
                    }
                }
                continue;
            }
#endif
            row_store_block_tail(m, n, k, A, B, C, i0, j0, ilen, jlen);
        }
    }
}

// Row-major large-output kernel for 4000x16000x128.
// C is written once, so aligned non-temporal stores are tested on the target node.
static void row_4000_16000_128_stream(const double* A, const double* B, double* C) {
#ifndef __AVX512F__
    row_tile_i8_j16(4000, 16000, 128, A, B, C);
#else
    constexpr int M = 4000;
    constexpr int N = 16000;
    constexpr int K = 128;
    constexpr int IB = 4;
    constexpr int JB = 40;
    constexpr int VEC = 5;
    constexpr int JGROUP = 14;
    constexpr int NBI = M / IB;
    constexpr int NBJ = N / JB;
    constexpr int NGROUP = (NBJ + JGROUP - 1) / JGROUP;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int group = 0; group < NGROUP; ++group) {
        for (int bi = 0; bi < NBI; ++bi) {
            const int i0 = bi * IB;
            const int bj_end = std::min(NBJ, (group + 1) * JGROUP);
            for (int bj = group * JGROUP; bj < bj_end; ++bj) {
                const int j0 = bj * JB;
                __m512d acc[IB][VEC];
                for (int ii = 0; ii < IB; ++ii) {
                    for (int v = 0; v < VEC; ++v) {
                        acc[ii][v] = _mm512_setzero_pd();
                    }
                }

                for (int l = 0; l < K; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * N + j0;
                    const __m512d bv0 = _mm512_load_pd(b);
                    const __m512d bv1 = _mm512_load_pd(b + 8);
                    const __m512d bv2 = _mm512_load_pd(b + 16);
                    const __m512d bv3 = _mm512_load_pd(b + 24);
                    const __m512d bv4 = _mm512_load_pd(b + 32);
                    const double* a = A + static_cast<std::size_t>(l) * M + i0;
                    for (int ii = 0; ii < IB; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc[ii][0] = _mm512_fmadd_pd(av, bv0, acc[ii][0]);
                        acc[ii][1] = _mm512_fmadd_pd(av, bv1, acc[ii][1]);
                        acc[ii][2] = _mm512_fmadd_pd(av, bv2, acc[ii][2]);
                        acc[ii][3] = _mm512_fmadd_pd(av, bv3, acc[ii][3]);
                        acc[ii][4] = _mm512_fmadd_pd(av, bv4, acc[ii][4]);
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
            }
        }
    }
    _mm_sfence();
#endif
}

// Row-major tiny-output kernel for 8x16x16000.
// Parallelism is along k, followed by a 128-value reduction.
static void row_8_16_16000_k_split(const double* A, const double* B, double* C) {
    constexpr int M = 8;
    constexpr int N = 16;
    constexpr int K = 16000;
    const int threads = limited_threads(8);
    std::vector<double> partial(static_cast<std::size_t>(threads) * M * N, 0.0);

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        int tid = 0;
        int nth = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nth = omp_get_num_threads();
#endif
        const int l0 = (K * tid) / nth;
        const int l1 = (K * (tid + 1)) / nth;
        double* out = partial.data() + static_cast<std::size_t>(tid) * M * N;

#ifdef __AVX512F__
        __m512d lo[M];
        __m512d hi[M];
        for (int i = 0; i < M; ++i) {
            lo[i] = _mm512_setzero_pd();
            hi[i] = _mm512_setzero_pd();
        }
        for (int l = l0; l < l1; ++l) {
            const double* a = A + static_cast<std::size_t>(l) * M;
            const double* b = B + static_cast<std::size_t>(l) * N;
            const __m512d b0 = _mm512_loadu_pd(b);
            const __m512d b1 = _mm512_loadu_pd(b + 8);
            for (int i = 0; i < M; ++i) {
                const __m512d av = _mm512_set1_pd(a[i]);
                lo[i] = _mm512_fmadd_pd(av, b0, lo[i]);
                hi[i] = _mm512_fmadd_pd(av, b1, hi[i]);
            }
        }
        for (int i = 0; i < M; ++i) {
            _mm512_storeu_pd(out + i * N, lo[i]);
            _mm512_storeu_pd(out + i * N + 8, hi[i]);
        }
#else
        for (int l = l0; l < l1; ++l) {
            const double* a = A + static_cast<std::size_t>(l) * M;
            const double* b = B + static_cast<std::size_t>(l) * N;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    out[i * N + j] += a[i] * b[j];
                }
            }
        }
#endif
    }

    for (int idx = 0; idx < M * N; ++idx) {
        double sum = 0.0;
        for (int t = 0; t < threads; ++t) {
            sum += partial[static_cast<std::size_t>(t) * M * N + idx];
        }
        C[idx] = sum;
    }
}

// Row-major small-k, large-n kernel for 32x16000x16.
static void row_32_16000_16_wide_j(const double* A, const double* B, double* C) {
    constexpr int M = 32;
    constexpr int N = 16000;
    constexpr int K = 16;
    constexpr int IB = 4;
    constexpr int JB = 32;
    const int threads = limited_threads(48);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int j0 = 0; j0 < N; j0 += JB) {
        for (int i0 = 0; i0 < M; i0 += IB) {
#ifdef __AVX512F__
            __m512d acc[IB][4];
            for (int ii = 0; ii < IB; ++ii) {
                for (int v = 0; v < 4; ++v) {
                    acc[ii][v] = _mm512_setzero_pd();
                }
            }
            for (int l = 0; l < K; ++l) {
                const double* b = B + static_cast<std::size_t>(l) * N + j0;
                const __m512d bv0 = _mm512_loadu_pd(b);
                const __m512d bv1 = _mm512_loadu_pd(b + 8);
                const __m512d bv2 = _mm512_loadu_pd(b + 16);
                const __m512d bv3 = _mm512_loadu_pd(b + 24);
                const double* a = A + static_cast<std::size_t>(l) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    const __m512d av = _mm512_set1_pd(a[ii]);
                    acc[ii][0] = _mm512_fmadd_pd(av, bv0, acc[ii][0]);
                    acc[ii][1] = _mm512_fmadd_pd(av, bv1, acc[ii][1]);
                    acc[ii][2] = _mm512_fmadd_pd(av, bv2, acc[ii][2]);
                    acc[ii][3] = _mm512_fmadd_pd(av, bv3, acc[ii][3]);
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
            row_store_block_tail(M, N, K, A, B, C, i0, j0, IB, JB);
#endif
        }
    }
}

// Row-major small-square kernel for 144x144x144.
static void row_144_144_144_limited(const double* A, const double* B, double* C) {
    constexpr int M = 144;
    constexpr int N = 144;
    constexpr int K = 144;
    constexpr int IB = 9;
    constexpr int JB = 24;
    const int threads = limited_threads(32);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
#endif
    for (int i0 = 0; i0 < M; i0 += IB) {
        for (int j0 = 0; j0 < N; j0 += JB) {
#ifdef __AVX512F__
            __m512d acc[IB][3];
            for (int ii = 0; ii < IB; ++ii) {
                for (int v = 0; v < 3; ++v) {
                    acc[ii][v] = _mm512_setzero_pd();
                }
            }
            for (int l = 0; l < K; ++l) {
                const double* b = B + static_cast<std::size_t>(l) * N + j0;
                const __m512d bv0 = _mm512_loadu_pd(b);
                const __m512d bv1 = _mm512_loadu_pd(b + 8);
                const __m512d bv2 = _mm512_loadu_pd(b + 16);
                const double* a = A + static_cast<std::size_t>(l) * M + i0;
                for (int ii = 0; ii < IB; ++ii) {
                    const __m512d av = _mm512_set1_pd(a[ii]);
                    acc[ii][0] = _mm512_fmadd_pd(av, bv0, acc[ii][0]);
                    acc[ii][1] = _mm512_fmadd_pd(av, bv1, acc[ii][1]);
                    acc[ii][2] = _mm512_fmadd_pd(av, bv2, acc[ii][2]);
                }
            }
            for (int ii = 0; ii < IB; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * N + j0;
                _mm512_storeu_pd(c, acc[ii][0]);
                _mm512_storeu_pd(c + 8, acc[ii][1]);
                _mm512_storeu_pd(c + 16, acc[ii][2]);
            }
#else
            row_store_block_tail(M, N, K, A, B, C, i0, j0, IB, JB);
#endif
        }
    }
}

static inline void col_i8_kernel(int ilen, int k,
                                 const double* A, const double* b, double* c) {
#ifdef __AVX512F__
    __m512d acc[8];
    for (int ii = 0; ii < ilen; ++ii) {
        acc[ii] = _mm512_setzero_pd();
    }
    int l = 0;
    for (; l + 7 < k; l += 8) {
        const __m512d bv = _mm512_loadu_pd(b + l);
        for (int ii = 0; ii < ilen; ++ii) {
            const double* a = A + static_cast<std::size_t>(ii) * k + l;
            acc[ii] = _mm512_fmadd_pd(_mm512_loadu_pd(a), bv, acc[ii]);
        }
    }
    for (int ii = 0; ii < ilen; ++ii) {
        const double* a = A + static_cast<std::size_t>(ii) * k;
        double sum = hsum_vec(acc[ii]);
        for (int lt = l; lt < k; ++lt) {
            sum += a[lt] * b[lt];
        }
        c[ii] = sum;
    }
#else
    double sum[8] = {};
    for (int l = 0; l < k; ++l) {
        const double bv = b[l];
        for (int ii = 0; ii < ilen; ++ii) {
            sum[ii] += A[static_cast<std::size_t>(ii) * k + l] * bv;
        }
    }
    for (int ii = 0; ii < ilen; ++ii) {
        c[ii] = sum[ii];
    }
#endif
}

static void col_dot_element_parallel(int m, int n, int k,
                                     const double* A, const double* B, double* C) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[static_cast<std::size_t>(j) * m + i] =
                dot_contiguous(A + static_cast<std::size_t>(i) * k,
                               B + static_cast<std::size_t>(j) * k,
                               k);
        }
    }
}

static void col_dot_i8_block(int m, int n, int k,
                             const double* A, const double* B, double* C,
                             bool collapse_i) {
    constexpr int IB = 8;

    if (collapse_i) {
        const int nbi = (m + IB - 1) / IB;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int j = 0; j < n; ++j) {
            for (int bi = 0; bi < nbi; ++bi) {
                const int i0 = bi * IB;
                const int ilen = std::min(IB, m - i0);
                const double* b = B + static_cast<std::size_t>(j) * k;
                double* c = C + static_cast<std::size_t>(j) * m + i0;
                col_i8_kernel(ilen, k, A + static_cast<std::size_t>(i0) * k, b, c);
            }
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        const double* b = B + static_cast<std::size_t>(j) * k;
        for (int i0 = 0; i0 < m; i0 += IB) {
            const int ilen = std::min(IB, m - i0);
            double* c = C + static_cast<std::size_t>(j) * m + i0;
            col_i8_kernel(ilen, k, A + static_cast<std::size_t>(i0) * k, b, c);
        }
    }
}

// Col-major large-output kernel for 4000x16000x128.
// Packing A changes repeated strided loads into unit-stride loads inside the j-block loop.
static void col_4000_16000_128_pack_a(const double* A, const double* B, double* C) {
#ifndef __AVX512F__
    col_dot_i8_block(4000, 16000, 128, A, B, C, false);
#else
    constexpr int M = 4000;
    constexpr int N = 16000;
    constexpr int K = 128;
    constexpr int IB = 16;
    constexpr int JB = 8;
    static std::vector<double> apack;
    if (apack.size() != static_cast<std::size_t>(K) * M) {
        apack.resize(static_cast<std::size_t>(K) * M);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int l = 0; l < K; ++l) {
        double* dst = apack.data() + static_cast<std::size_t>(l) * M;
        for (int i = 0; i < M; ++i) {
            dst[i] = A[static_cast<std::size_t>(i) * K + l];
        }
    }

    // Keep a whole group of B columns on one thread.  The old collapse(2)
    // schedule split (j0,i0) pairs across threads, which reloaded the same
    // 8x128 B panel for every i-block and created finer scheduling work.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j0 = 0; j0 < N; j0 += JB) {
        for (int i0 = 0; i0 < M; i0 += IB) {
            __m512d lo[JB];
            __m512d hi[JB];
            for (int x = 0; x < JB; ++x) {
                lo[x] = _mm512_setzero_pd();
                hi[x] = _mm512_setzero_pd();
            }
            for (int l = 0; l < K; ++l) {
                const double* a = apack.data() + static_cast<std::size_t>(l) * M + i0;
                const __m512d av0 = _mm512_loadu_pd(a);
                const __m512d av1 = _mm512_loadu_pd(a + 8);
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
        }
    }
    _mm_sfence();
#endif
}

// Col-major 32x16000x16: A fits in L1 after transposition, then each B column is cheap.
// The four-column AVX-512 path is deliberately unrolled to reduce loop and
// indexed-register overhead on this small-k case.
static void col_32_16000_16_multi_col(const double* A, const double* B, double* C) {
    constexpr int M = 32;
    constexpr int N = 16000;
    constexpr int K = 16;
    constexpr int JB = 4;
    const int threads = limited_threads(24);
    alignas(64) double at[K][M];
    for (int l = 0; l < K; ++l) {
        for (int i = 0; i < M; ++i) {
            at[l][i] = A[static_cast<std::size_t>(i) * K + l];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int j0 = 0; j0 < N; j0 += JB) {
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
        for (int x = 0; x < JB; ++x) {
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

static void col_144_144_144_two_cols(const double* A, const double* B, double* C) {
    constexpr int M = 144;
    constexpr int N = 144;
    constexpr int K = 144;
    constexpr int IB = 8;
    const int threads = limited_threads(32);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int j = 0; j < N; j += 2) {
        const double* b0 = B + static_cast<std::size_t>(j) * K;
        const double* b1 = B + static_cast<std::size_t>(j + 1) * K;
        for (int i0 = 0; i0 < M; i0 += IB) {
#ifdef __AVX512F__
            __m512d acc0[IB];
            __m512d acc1[IB];
            for (int ii = 0; ii < IB; ++ii) {
                acc0[ii] = _mm512_setzero_pd();
                acc1[ii] = _mm512_setzero_pd();
            }
            for (int l = 0; l < K; l += 8) {
                const __m512d bv0 = _mm512_loadu_pd(b0 + l);
                const __m512d bv1 = _mm512_loadu_pd(b1 + l);
                for (int ii = 0; ii < IB; ++ii) {
                    const double* a = A + static_cast<std::size_t>(i0 + ii) * K + l;
                    const __m512d av = _mm512_loadu_pd(a);
                    acc0[ii] = _mm512_fmadd_pd(av, bv0, acc0[ii]);
                    acc1[ii] = _mm512_fmadd_pd(av, bv1, acc1[ii]);
                }
            }
            double* c0 = C + static_cast<std::size_t>(j) * M + i0;
            double* c1 = C + static_cast<std::size_t>(j + 1) * M + i0;
            for (int ii = 0; ii < IB; ++ii) {
                c0[ii] = hsum_vec(acc0[ii]);
                c1[ii] = hsum_vec(acc1[ii]);
            }
#else
            double* c0 = C + static_cast<std::size_t>(j) * M + i0;
            double* c1 = C + static_cast<std::size_t>(j + 1) * M + i0;
            for (int ii = 0; ii < IB; ++ii) {
                const double* a = A + static_cast<std::size_t>(i0 + ii) * K;
                double sum0 = 0.0;
                double sum1 = 0.0;
                for (int l = 0; l < K; ++l) {
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

static void tsmm_opt_row(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m == 4000 && n == 16000 && k == 128) {
        row_4000_16000_128_stream(A, B, C);
    } else if (m == 8 && n == 16 && k == 16000) {
        row_8_16_16000_k_split(A, B, C);
    } else if (m == 32 && n == 16000 && k == 16) {
        row_32_16000_16_wide_j(A, B, C);
    } else if (m == 144 && n == 144 && k == 144) {
        row_144_144_144_limited(A, B, C);
    } else {
        row_tile_i8_j16(m, n, k, A, B, C);
    }
}

static void tsmm_opt_col(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m == 4000 && n == 16000 && k == 128) {
        col_4000_16000_128_pack_a(A, B, C);
    } else if (m == 32 && n == 16000 && k == 16) {
        col_32_16000_16_multi_col(A, B, C);
    } else if (m == 144 && n == 144 && k == 144) {
        col_144_144_144_two_cols(A, B, C);
    } else if (m <= 16 && n <= 64 && k >= 1024) {
        col_dot_element_parallel(m, n, k, A, B, C);
    } else {
        col_dot_i8_block(m, n, k, A, B, C, n < 512);
    }
}

void tsmm_opt(int m, int n, int k,
              const double* A, const double* B, double* C,
              Layout layout) {
    if (layout == Layout::RowMajor) {
        tsmm_opt_row(m, n, k, A, B, C);
    } else {
        tsmm_opt_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("opt", tsmm_opt);
