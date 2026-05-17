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

static inline void row_fma_segment(double* c, const double* b, double av, int len) {
    int j = 0;
#ifdef __AVX512F__
    const __m512d avec = _mm512_set1_pd(av);
    for (; j + 31 < len; j += 32) {
        __m512d c0 = _mm512_loadu_pd(c + j);
        __m512d c1 = _mm512_loadu_pd(c + j + 8);
        __m512d c2 = _mm512_loadu_pd(c + j + 16);
        __m512d c3 = _mm512_loadu_pd(c + j + 24);
        c0 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j), c0);
        c1 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j + 8), c1);
        c2 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j + 16), c2);
        c3 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j + 24), c3);
        _mm512_storeu_pd(c + j, c0);
        _mm512_storeu_pd(c + j + 8, c1);
        _mm512_storeu_pd(c + j + 16, c2);
        _mm512_storeu_pd(c + j + 24, c3);
    }
    for (; j + 7 < len; j += 8) {
        __m512d cv = _mm512_loadu_pd(c + j);
        cv = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j), cv);
        _mm512_storeu_pd(c + j, cv);
    }
#endif
    for (; j < len; ++j) {
        c[j] += av * b[j];
    }
}

static void row_tiny_output_large_k(int m, int n, int k,
                                    const double* A, const double* B, double* C) {
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    const int out_size = m * n;
    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

#ifdef _OPENMP
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;
#pragma omp for schedule(static)
        for (int l = 0; l < k; ++l) {
            const double* a = A + static_cast<std::size_t>(l) * m;
            const double* b = B + static_cast<std::size_t>(l) * n;
            for (int i = 0; i < m; ++i) {
                row_fma_segment(ct + static_cast<std::size_t>(i) * n, b, a[i], n);
            }
        }
    }
#else
    double* ct = tmp[0].data();
    for (int l = 0; l < k; ++l) {
        const double* a = A + static_cast<std::size_t>(l) * m;
        const double* b = B + static_cast<std::size_t>(l) * n;
        for (int i = 0; i < m; ++i) {
            row_fma_segment(ct + static_cast<std::size_t>(i) * n, b, a[i], n);
        }
    }
#endif

    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    for (int idx = 0; idx < out_size; ++idx) {
        double sum = 0.0;
        for (int t = 0; t < nthreads; ++t) {
            sum += tmp[static_cast<std::size_t>(t) * out_size + idx];
        }
        C[idx] = sum;
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
                        for (int jj = 8; jj < jlen; ++jj) {
                            double sum = 0.0;
                            for (int l = 0; l < k; ++l) {
                                sum += A[static_cast<std::size_t>(l) * m + i0 + ii] *
                                       B[static_cast<std::size_t>(l) * n + j0 + jj];
                            }
                            c[jj] = sum;
                        }
                    }
                }
                continue;
            }
#endif

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
    }
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

static void tsmm_opt_row(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m == 8 && n == 16 && k == 16000) {
        row_tiny_output_large_k(m, n, k, A, B, C);
    } else {
        row_tile_i8_j16(m, n, k, A, B, C);
    }
}

static void tsmm_opt_col(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m <= 16 && n <= 64 && k >= 1024) {
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
