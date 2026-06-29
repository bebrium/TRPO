#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "cblas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using Clock = chrono::steady_clock;

// -------------------------------------------------------------------
//  Вспомогательные функции
// -------------------------------------------------------------------
template <typename T>
bool almost_equal(T lhs, T rhs, T eps = static_cast<T>(1e-5)) {
    T diff = abs(lhs - rhs);
    T scale = max({T(1), abs(lhs), abs(rhs)});
    return diff <= eps * scale;
}

template <typename T>
bool matrices_close(const vector<T>& A, const vector<T>& B, T eps) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i)
        if (!almost_equal(A[i], B[i], eps)) return false;
    return true;
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
string type_name() {
    return is_same_v<T, float> ? "float" : "double";
}

// -------------------------------------------------------------------
//  Ручная реализация TRMM (последовательная)
//  Вычисляет: B := alpha * op(A) * B   (если Side == CblasLeft)
//  или        B := alpha * B * op(A)   (если Side == CblasRight)
//  A – треугольная матрица (Upper/Lower, Unit/NonUnit, Trans/NoTrans)
// -------------------------------------------------------------------
template <typename T>
void manual_trmm_seq(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                     CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
                     int M, int N, T alpha,
                     const T* A, int lda,
                     T* B, int ldb) {
    // Для простоты поддерживаем только RowMajor (как и в OpenBLAS-тесте)
    if (order != CblasRowMajor) {
        cerr << "Unsupported order in manual_trmm_seq\n";
        return;
    }

    auto is_unit = (diag == CblasUnit);
    auto is_upper = (uplo == CblasUpper);
    auto is_trans = (transA == CblasTrans);

    if (side == CblasLeft) {
        // B := alpha * A * B   (M строк A, N столбцов B)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0;
                if (is_trans) {
                    // A^T: A[k,i] для k=0..M-1, но треугольность по k,i
                    for (int k = 0; k < M; ++k) {
                        if (is_upper) { // A верхняя треугольная в A^T => нижняя
                            if (k > i) continue;
                        } else { // lower
                            if (k < i) continue;
                        }
                        T a_ik = (k == i && is_unit) ? T(1) : A[k * lda + i];
                        sum += a_ik * B[k * ldb + j];
                    }
                } else {
                    // A * B
                    for (int k = 0; k < M; ++k) {
                        if (is_upper) { // верхняя: только k <= i
                            if (k > i) continue;
                        } else { // нижняя: только k >= i
                            if (k < i) continue;
                        }
                        T a_ik = (k == i && is_unit) ? T(1) : A[i * lda + k];
                        sum += a_ik * B[k * ldb + j];
                    }
                }
                B[i * ldb + j] = alpha * sum;
            }
        }
    } else { // Right
        // B := alpha * B * A
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0;
                if (is_trans) {
                    // B * A^T
                    for (int k = 0; k < N; ++k) {
                        if (is_upper) { // A верхняя => A^T нижняя
                            if (k < j) continue;
                        } else {
                            if (k > j) continue;
                        }
                        T a_jk = (j == k && is_unit) ? T(1) : A[j * lda + k];
                        sum += B[i * ldb + k] * a_jk;
                    }
                } else {
                    // B * A
                    for (int k = 0; k < N; ++k) {
                        if (is_upper) {
                            if (k > j) continue;
                        } else {
                            if (k < j) continue;
                        }
                        T a_kj = (j == k && is_unit) ? T(1) : A[k * lda + j];
                        sum += B[i * ldb + k] * a_kj;
                    }
                }
                B[i * ldb + j] = alpha * sum;
            }
        }
    }
}

// Параллельная версия с OpenMP (только для Left, Upper, NoTrans, NonUnit для простоты)
template <typename T>
void manual_trmm_par(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                     CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
                     int M, int N, T alpha,
                     const T* A, int lda,
                     T* B, int ldb) {
    if (order != CblasRowMajor) return;
    if (side == CblasLeft && uplo == CblasUpper && transA == CblasNoTrans && diag == CblasNonUnit) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0;
                for (int k = 0; k <= i; ++k) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                B[i * ldb + j] = alpha * sum;
            }
        }
#else
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0;
                for (int k = 0; k <= i; ++k) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                B[i * ldb + j] = alpha * sum;
            }
        }
#endif
    } else {
        manual_trmm_seq(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
    }
}

// -------------------------------------------------------------------
//  Проверка корректности (сравнение ручной и OpenBLAS)
// -------------------------------------------------------------------
template <typename T>
bool test_trmm() {
    const int M = 30, N = 25;
    T alpha = T(1.23);
    vector<T> A(M * M, 0);
    vector<T> B1(M * N, 0);
    vector<T> B2(M * N, 0);

    fill_random(A, T(-1), T(1));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < i; ++j)
            A[i * M + j] = 0;
    fill_random(B1, T(-2), T(2));
    B2 = B1;

    manual_trmm_seq<T>(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                       M, N, alpha, A.data(), M, B1.data(), N);

    if constexpr (is_same_v<T, float>) {
        cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    M, N, alpha, A.data(), M, B2.data(), N);
    } else {
        cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    M, N, alpha, A.data(), M, B2.data(), N);
    }

    return matrices_close(B1, B2, T(1e-4));
}

// -------------------------------------------------------------------
//  Измерение производительности
// -------------------------------------------------------------------
double elapsed_ms(const Clock::time_point& start, const Clock::time_point& finish) {
    return chrono::duration<double, milli>(finish - start).count();
}

double geometric_mean(const vector<double>& values) {
    if (values.empty()) return 0.0;
    double log_sum = 0.0;
    for (double v : values) log_sum += log(max(v, 1e-12));
    return exp(log_sum / values.size());
}

// Исправленная функция bench_trmm с двумя разными типами функций
template <typename T, typename Func1, typename Func2>
void bench_trmm(const string& label, Func1 manual_func, Func2 blas_func,
                int repeat = 10) {
    vector<double> manual_times, blas_times;
    manual_times.reserve(repeat);
    blas_times.reserve(repeat);

    for (int i = 0; i < repeat; ++i) {
        auto start = Clock::now();
        manual_func();
        manual_times.push_back(elapsed_ms(start, Clock::now()));

        start = Clock::now();
        blas_func();
        blas_times.push_back(elapsed_ms(start, Clock::now()));
    }

    double gm_manual = geometric_mean(manual_times);
    double gm_blas   = geometric_mean(blas_times);
    double percent = (gm_blas / gm_manual) * 100.0;

    cout << left << setw(25) << label
         << right << fixed << setprecision(3) << setw(14) << gm_manual
         << setw(14) << gm_blas
         << setw(14) << percent << "%\n";
}

// -------------------------------------------------------------------
//  Запуск тестов производительности для разных потоков
// -------------------------------------------------------------------
template <typename T>
void run_benchmark_for_type(int M, int N, int lda, int ldb, T alpha,
                            const vector<int>& thread_counts) {
    vector<T> A(M * lda, 0);
    vector<T> B(M * ldb, 0);

    fill_random(A, T(-1), T(1));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < i; ++j)
            A[i * lda + j] = 0;

    fill_random(B, T(-2), T(2));

    cout << "\n=== Тип: " << type_name<T>() << " | Размер матриц: " << M << "x" << N
         << " (A: " << M << "x" << M << ", B: " << M << "x" << N << ")"
         << " | alpha = " << alpha << "\n";
    cout << left << setw(25) << "Потоки / Реализация"
         << right << setw(14) << "Manual, ms"
         << setw(14) << "OpenBLAS, ms"
         << setw(14) << "% от OpenBLAS\n";
    cout << string(67, '-') << '\n';

    for (int threads : thread_counts) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        if (threads > 1) continue;
#endif
        string thread_label = (threads == 1) ? "1 поток (посл.)" : to_string(threads) + " потоков (паралл.)";

        auto manual_func = [&]() {
            vector<T> Btmp = B;
            if (threads == 1) {
                manual_trmm_seq<T>(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                   M, N, alpha, A.data(), lda, Btmp.data(), ldb);
            } else {
                manual_trmm_par<T>(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                   M, N, alpha, A.data(), lda, Btmp.data(), ldb);
            }
            volatile T dummy = Btmp[0];
        };

        auto blas_func = [&]() {
            vector<T> Btmp = B;
            if constexpr (is_same_v<T, float>) {
                cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                            M, N, alpha, A.data(), lda, Btmp.data(), ldb);
            } else {
                cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                            M, N, alpha, A.data(), lda, Btmp.data(), ldb);
            }
            volatile T dummy = Btmp[0];
        };

        bench_trmm<T>(thread_label, manual_func, blas_func, 10);
    }
}

// -------------------------------------------------------------------
//  Main
// -------------------------------------------------------------------
int main() {
    cout << "Сравнение производительности TRMM (C = alpha * A * B)\n";
    cout << "OpenBLAS принят за 100%\n";
#ifdef _OPENMP
    cout << "OpenMP доступен, макс. потоков: " << omp_get_max_threads() << "\n";
#else
    cout << "OpenMP НЕ доступен – параллельная версия будет последовательной\n";
#endif

    bool ok_float = test_trmm<float>();
    bool ok_double = test_trmm<double>();
    cout << "\nПроверка корректности:\n"
         << "  float  : " << (ok_float ? "OK" : "FAIL") << "\n"
         << "  double : " << (ok_double ? "OK" : "FAIL") << "\n";
    if (!ok_float || !ok_double) {
        cerr << "Ошибка: ручная реализация не совпадает с OpenBLAS.\n";
        return 1;
    }

    // Подберите размер так, чтобы время выполнения было около 1 минуты
    const int M = 5000;
    const int N = 5000;
    const float alpha_f = 1.5f;
    const double alpha_d = 1.5;

    vector<int> thread_counts = {1, 2, 4, 8, 16};

    run_benchmark_for_type<float>(M, N, M, N, alpha_f, thread_counts);
    run_benchmark_for_type<double>(M, N, M, N, alpha_d, thread_counts);

    cout << "\nТестирование завершено.\n";
    return 0;
}