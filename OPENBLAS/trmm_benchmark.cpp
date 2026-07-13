#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>

#include "cblas.h"
#include "openblas_config.h"

extern "C" void openblas_set_num_threads(int num_threads);
extern "C" int openblas_get_num_threads();

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using Clock = chrono::steady_clock;


template <typename T>
bool almost_equal(T a, T b, T eps = 1e-3) {
    T diff = abs(a - b);
    T scale = max({T(1), abs(a), abs(b)});
    return diff <= eps * scale;
}

template <typename T>
void fill_random(vector<T>& v, T low, T high) {
    static mt19937 rng(12345);
    if constexpr (is_same_v<T, float>) {
        uniform_real_distribution<float> dist(low, high);
        for (auto& x : v) x = dist(rng);
    } else {
        uniform_real_distribution<double> dist(low, high);
        for (auto& x : v) x = dist(rng);
    }
}


template <typename T>
void my_trmm_parallel(CBLAS_SIDE side, CBLAS_UPLO uplo,
                      CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
                      int M, int N, T alpha,
                      const T* A, int lda,
                      T* B, int ldb) {
    const bool left = (side == CblasLeft);
    const bool upper = (uplo == CblasUpper);
    const bool trans = (transA == CblasTrans);
    const bool unit = (diag == CblasUnit);

    vector<T> B_copy(M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            B_copy[i * N + j] = B[i * ldb + j];

    if (left) {
        if (!trans) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = i; k < M; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[i * lda + k];
                            sum += a_val * B_copy[k * N + j];
                        }
                    } else {
                        for (int k = 0; k <= i; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[i * lda + k];
                            sum += a_val * B_copy[k * N + j];
                        }
                    }
                    B[i * ldb + j] = alpha * sum;
                }
            }
        } else {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = 0; k <= i; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[k * lda + i];
                            sum += a_val * B_copy[k * N + j];
                        }
                    } else {
                        for (int k = i; k < M; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[k * lda + i];
                            sum += a_val * B_copy[k * N + j];
                        }
                    }
                    B[i * ldb + j] = alpha * sum;
                }
            }
        }
    } else {
        if (!trans) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = 0; k <= j; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[k * lda + j];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    } else {
                        for (int k = j; k < N; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[k * lda + j];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    }
                    B[i * ldb + j] = alpha * sum;
                }
            }
        } else {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = j; k < N; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[j * lda + k];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    } else {
                        for (int k = 0; k <= j; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[j * lda + k];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    }
                    B[i * ldb + j] = sum;
                }
            }
        }
    }
}


template <typename T>
void my_trmm_sequential(CBLAS_SIDE side, CBLAS_UPLO uplo,
                        CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
                        int M, int N, T alpha,
                        const T* A, int lda,
                        T* B, int ldb) {
    const bool left = (side == CblasLeft);
    const bool upper = (uplo == CblasUpper);
    const bool trans = (transA == CblasTrans);
    const bool unit = (diag == CblasUnit);

    vector<T> B_copy(M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            B_copy[i * N + j] = B[i * ldb + j];

    if (left) {
        if (!trans) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = i; k < M; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[i * lda + k];
                            sum += a_val * B_copy[k * N + j];
                        }
                    } else {
                        for (int k = 0; k <= i; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[i * lda + k];
                            sum += a_val * B_copy[k * N + j];
                        }
                    }
                    B[i * ldb + j] = alpha * sum;
                }
            }
        } else {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = 0; k <= i; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[k * lda + i];
                            sum += a_val * B_copy[k * N + j];
                        }
                    } else {
                        for (int k = i; k < M; ++k) {
                            T a_val = (unit && k == i) ? T(1) : A[k * lda + i];
                            sum += a_val * B_copy[k * N + j];
                        }
                    }
                    B[i * ldb + j] = alpha * sum;
                }
            }
        }
    } else {
        if (!trans) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = 0; k <= j; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[k * lda + j];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    } else {
                        for (int k = j; k < N; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[k * lda + j];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    }
                    B[i * ldb + j] = alpha * sum;
                }
            }
        } else {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    T sum = 0;
                    if (upper) {
                        for (int k = j; k < N; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[j * lda + k];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    } else {
                        for (int k = 0; k <= j; ++k) {
                            T a_val = (unit && k == j) ? T(1) : A[j * lda + k];
                            sum += B_copy[i * N + k] * a_val;
                        }
                    }
                    B[i * ldb + j] = alpha * sum;
                }
            }
        }
    }
}


template <typename T>
void my_trmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
             CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
             int M, int N, T alpha,
             const T* A, int lda,
             T* B, int ldb, bool parallel = false) {
    if (order != CblasRowMajor) {
        cerr << "Only CblasRowMajor supported\n";
        return;
    }
    
    if (parallel) {
        my_trmm_parallel<T>(side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
    } else {
        my_trmm_sequential<T>(side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
    }
}


template <typename T>
bool test_all_variants() {
    const int M = 30, N = 25;
    const T alpha = T(1.23);
    
    auto side_name = [](CBLAS_SIDE s) -> const char* {
        return (s == CblasLeft) ? "Left" : "Right";
    };
    auto uplo_name = [](CBLAS_UPLO u) -> const char* {
        return (u == CblasUpper) ? "Upper" : "Lower";
    };
    auto trans_name = [](CBLAS_TRANSPOSE t) -> const char* {
        return (t == CblasNoTrans) ? "NoTrans" : "Trans";
    };
    auto diag_name = [](CBLAS_DIAG d) -> const char* {
        return (d == CblasNonUnit) ? "NonUnit" : "Unit";
    };

    vector<CBLAS_SIDE> sides = {CblasLeft, CblasRight};
    vector<CBLAS_UPLO> uplos = {CblasUpper, CblasLower};
    vector<CBLAS_TRANSPOSE> trans = {CblasNoTrans, CblasTrans};
    vector<CBLAS_DIAG> diags = {CblasNonUnit, CblasUnit};

    for (auto side : sides) {
        for (auto uplo : uplos) {
            for (auto tr : trans) {
                for (auto diag : diags) {
                    int m = M;
                    int n = N;
                    int dimA = (side == CblasLeft) ? m : n;
                    
                    vector<T> A(dimA * dimA, 0);
                    vector<T> B_ref(m * n, 0);
                    vector<T> B_my(m * n, 0);

                    fill_random(A, T(-1), T(1));
                    if (uplo == CblasUpper) {
                        for (int i = 0; i < dimA; ++i)
                            for (int j = 0; j < i; ++j)
                                A[i * dimA + j] = T(0);
                    } else {
                        for (int i = 0; i < dimA; ++i)
                            for (int j = i + 1; j < dimA; ++j)
                                A[i * dimA + j] = T(0);
                    }
                    
                    if (diag == CblasUnit) {
                        for (int i = 0; i < dimA; ++i)
                            A[i * dimA + i] = T(1);
                    }
                    
                    fill_random(B_ref, T(-2), T(2));
                    B_my = B_ref;

                    if constexpr (is_same_v<T, float>) {
                        cblas_strmm(CblasRowMajor, side, uplo, tr, diag,
                                   m, n, alpha, A.data(), dimA, B_ref.data(), n);
                    } else {
                        cblas_dtrmm(CblasRowMajor, side, uplo, tr, diag,
                                   m, n, alpha, A.data(), dimA, B_ref.data(), n);
                    }

                    my_trmm_sequential<T>(side, uplo, tr, diag,
                                         m, n, alpha, A.data(), dimA, B_my.data(), n);

                    for (size_t i = 0; i < B_ref.size(); ++i) {
                        if (!almost_equal(B_ref[i], B_my[i], T(1e-3))) {
                            cerr << "Failed: " << side_name(side) << " "
                                 << uplo_name(uplo) << " "
                                 << trans_name(tr) << " "
                                 << diag_name(diag) << endl;
                            return false;
                        }
                    }
                }
            }
        }
    }
    
    return true;
}


double elapsed_ms(const Clock::time_point& start, const Clock::time_point& finish) {
    return chrono::duration<double, milli>(finish - start).count();
}

double geometric_mean(const vector<double>& vals) {
    if (vals.empty()) return 0.0;
    double log_sum = 0.0;
    for (double v : vals) log_sum += log(v);
    return exp(log_sum / vals.size());
}


template <typename T>
int find_best_matrix_size(int target_seconds, T alpha, int num_runs = 10) {
    int sizes[] = {1024, 1228, 1473, 1767, 2120, 2544, 3052, 3662, 4394, 5273, 6328, 7594, 9113};
    int best_size = 1024;
    double best_time = 0;
    
    cout << "Подбор размера матрицы для ~" << target_seconds << " сек (10 запусков)...\n";
    
    for (int size : sizes) {
        int M = size, N = size;
        vector<T> A(M * M, 0);
        vector<T> B(M * N, 0);
        
        fill_random(A, T(-1), T(1));
        for (int i = 0; i < M; ++i)
            for (int j = i + 1; j < M; ++j)
                A[i * M + j] = T(0);
        fill_random(B, T(-2), T(2));

        auto start = Clock::now();
        my_trmm_sequential<T>(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                              M, N, alpha, A.data(), M, B.data(), N);
        double one_run = elapsed_ms(start, Clock::now()) / 1000.0;
        double total_time = one_run * num_runs;
        
        cout << "  " << size << "x" << size << ": " << fixed << setprecision(3) 
             << one_run << " сек (1 запуск), " << total_time << " сек";
        
        if (fabs(total_time - target_seconds) < fabs(best_time - target_seconds) || best_time == 0) {
            best_size = size;
            best_time = total_time;
            cout << " ✓";
        }
        cout << endl;
        
        if (total_time > target_seconds) break;
    }
    
    cout << "\nВыбран размер: " << best_size << "x" << best_size 
         << " (общее время: " << fixed << setprecision(3) << best_time << " сек)\n";
    return best_size;
}

template <typename T>
void run_benchmark(int M, int N, int max_threads, T alpha, int num_runs = 10) {
    cout << "\nПодготовка данных для " << (is_same_v<T, float> ? "float" : "double") 
         << " (" << M << "x" << M << ")...\n";
    
    vector<T> A(M * M, 0);
    vector<T> B(M * N, 0);
    
    fill_random(A, T(-1), T(1));
    for (int i = 0; i < M; ++i)
        for (int j = i + 1; j < M; ++j)
            A[i * M + j] = T(0);
    fill_random(B, T(-2), T(2));

    vector<int> thread_counts;
    for (int t : {1, 2, 4, 8, 16}) {
        if (t <= max_threads) thread_counts.push_back(t);
    }

    cout << "\n=== " << (is_same_v<T, float> ? "FLOAT" : "DOUBLE") 
         << " | N=" << M << " | alpha=" << alpha << " | " << num_runs << " runs ===\n";
    cout << left << setw(10) << "Threads"
         << right << setw(16) << "My (ms)"
         << setw(16) << "OpenBLAS (ms)"
         << setw(18) << "Perf %" << "\n";
    cout << string(60, '-') << '\n';

    for (int nth : thread_counts) {
        openblas_set_num_threads(nth);
        
        vector<double> my_times, blas_times;
        my_times.reserve(num_runs);
        blas_times.reserve(num_runs);
        
        cout << nth << " threads... " << flush;

        for (int run = 0; run < num_runs; ++run) {
            vector<T> B_my = B;
            vector<T> B_blas = B;

            auto start = Clock::now();
            my_trmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    M, N, alpha, A.data(), M, B_my.data(), N, nth > 1);
            my_times.push_back(elapsed_ms(start, Clock::now()));

            start = Clock::now();
            if constexpr (is_same_v<T, float>) {
                cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                           M, N, alpha, A.data(), M, B_blas.data(), N);
            } else {
                cblas_dtrmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                           M, N, alpha, A.data(), M, B_blas.data(), N);
            }
            blas_times.push_back(elapsed_ms(start, Clock::now()));
        }

        double my_geom = geometric_mean(my_times);
        double blas_geom = geometric_mean(blas_times);
        double perf = (blas_geom / my_geom) * 100.0;

        cout << "done\n";
        cout << left << setw(10) << nth
             << right << fixed << setprecision(2) << setw(16) << my_geom
             << setw(16) << blas_geom
             << setw(17) << perf << " %\n";
    }
}


int main() {
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
    cout << "OpenMP threads available: " << max_threads << "\n";
#endif


    bool ok_float = test_all_variants<float>();
    bool ok_double = test_all_variants<double>();
    
    
    if (!ok_float || !ok_double) {
        cerr << "\nERROR: Implementation failed correctness tests!\n";
        return 1;
    }

    const int TARGET_SECONDS = 60;
    const int NUM_RUNS = 10;


    int M = find_best_matrix_size<float>(TARGET_SECONDS, 1.5f, NUM_RUNS);
    int N = M;


    cout << "Размер матрицы: " << M << "x" << M << "\n";


    run_benchmark<float>(M, N, max_threads, 1.5f, NUM_RUNS);
    run_benchmark<double>(M, N, max_threads, 1.5, NUM_RUNS);

    cout << "\nBenchmark completed.\n";
    return 0;
}