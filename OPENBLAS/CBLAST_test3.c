#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include <signal.h>
#include <setjmp.h>

#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define BLUE    "\033[34m"


static jmp_buf env_buffer;
static volatile sig_atomic_t got_signal = 0;

void segv_handler(int sig) {
    got_signal = 1;
    longjmp(env_buffer, 1); 
}

typedef struct {
    int passed;
    int total;
} TestSuite;

void ts_init(TestSuite* ts) { ts->passed = 0; ts->total = 0; }

void ts_report(TestSuite* ts, int result, const char* name) {
    ts->total++;
    if (result) {
        ts->passed++;
        printf(GREEN "  [OK]   " RESET "%s\n", name);
    } else {
        printf(RED "  [FAIL] " RESET "%s (Сбой или неверный результат)\n", name);
    }
}

void ts_summary(TestSuite* ts) {
    printf("\n" CYAN "=== ИТОГИ ТЕСТИРОВАНИЯ ===" RESET "\n");
    printf("Всего тестов: %d\n", ts->total);
    printf("Успешно:      " GREEN "%d" RESET "\n", ts->passed);
    
    if (ts->passed == ts->total) {
        printf(GREEN "ПОЛНОЕ ПОКРЫТИЕ ВСЕХ ФУНКЦИЙ!" RESET "\n");
    } else {
        printf(RED "Ошибок:       %d" RESET "\n", ts->total - ts->passed);
    }
}

void init_f(float* p, int n, float val) { for(int i=0; i<n; i++) p[i] = val; }
void init_d(double* p, int n, double val) { for(int i=0; i<n; i++) p[i] = val; }
void init_c(float complex* p, int n, float complex val) { for(int i=0; i<n; i++) p[i] = val; }
void init_z(double complex* p, int n, double complex val) { for(int i=0; i<n; i++) p[i] = val; }

void fill_seq_f(float* m, int r, int c, int ld) {
    for(int i=0; i<r; i++) for(int j=0; j<c; j++) m[i*ld+j] = (float)(i*c + j + 1);
}
void fill_seq_d(double* m, int r, int c, int ld) {
    for(int i=0; i<r; i++) for(int j=0; j<c; j++) m[i*ld+j] = (double)(i*c + j + 1);
}
void fill_seq_c(float complex* m, int r, int c, int ld) {
    for(int i=0; i<r; i++) for(int j=0; j<c; j++) m[i*ld+j] = (float)(i*c + j + 1) + 0.0f*I;
}
void fill_seq_z(double complex* m, int r, int c, int ld) {
    for(int i=0; i<r; i++) for(int j=0; j<c; j++) m[i*ld+j] = (double)(i*c + j + 1) + 0.0*I;
}

int check_f(float* m, int n) { for(int i=0; i<n; i++) if(!isfinite(m[i])) return 0; return 1; }
int check_d(double* m, int n) { for(int i=0; i<n; i++) if(!isfinite(m[i])) return 0; return 1; }

#define CPTR(p) ((void*)(p))

int run_safe(int (*func)(void)) {
    got_signal = 0;
    signal(SIGSEGV, segv_handler);
    signal(SIGABRT, segv_handler);
    if (setjmp(env_buffer) == 0) return func();
    return 0;
}

int t_sgemm() {
    float A[9], B[9], C[9]; 

    fill_seq_f(A,3,3,3); fill_seq_f(B,3,3,3); fill_seq_f(C,3,3,3);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3,3,3, 1.0f, A, 3, B, 3, 0.0f, C, 3);
    return check_f(C, 9);
}
int t_dgemm() {
    double A[9], B[9], C[9];
    fill_seq_d(A,3,3,3); fill_seq_d(B,3,3,3); fill_seq_d(C,3,3,3);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 3,3,3, 1.0, A, 3, B, 3, 0.0, C, 3);
    return check_d(C, 9);
}
int t_cgemm() {
    float complex alpha=1.0f, beta=0.0f;
    float complex A[9], B[9], C[9];
    fill_seq_c(A,3,3,3); fill_seq_c(B,3,3,3); fill_seq_c(C,3,3,3);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, 3,3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}
int t_zgemm() {
    double complex alpha=1.0, beta=0.0;
    double complex A[9], B[9], C[9];
    fill_seq_z(A,3,3,3); fill_seq_z(B,3,3,3); fill_seq_z(C,3,3,3);
    cblas_zgemm(CblasColMajor, CblasTrans, CblasConjTrans, 3,3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}

int t_ssymm() {
    float A[9], B[9], C[9]; init_f(A,9,1.0f); fill_seq_f(B,3,3,3); fill_seq_f(C,3,3,3);
    cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, 3,3, 1.0f, A, 3, B, 3, 0.0f, C, 3);
    return check_f(C, 9);
}
int t_dsymm() {
    double A[9], B[9], C[9]; init_d(A,9,1.0); fill_seq_d(B,3,3,3); fill_seq_d(C,3,3,3);
    cblas_dsymm(CblasColMajor, CblasRight, CblasLower, 3,3, 1.0, A, 3, B, 3, 0.0, C, 3);
    return check_d(C, 9);
}
int t_csymm() {
    float complex alpha=1.0f, beta=0.0f;
    float complex A[9], B[9], C[9]; init_c(A,9,1.0f); fill_seq_c(B,3,3,3); fill_seq_c(C,3,3,3);
    cblas_csymm(CblasRowMajor, CblasLeft, CblasUpper, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}
int t_zsymm() {
    double complex alpha=1.0, beta=0.0;
    double complex A[9], B[9], C[9]; init_z(A,9,1.0); fill_seq_z(B,3,3,3); fill_seq_z(C,3,3,3);
    cblas_zsymm(CblasColMajor, CblasRight, CblasLower, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}

int t_ssyrk() {
    float A[9], C[9]; fill_seq_f(A,3,3,3);
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, 3,3, 1.0f, A, 3, 0.0f, C, 3);
    return check_f(C, 9);
}
int t_dsyrk() {
    double A[9], C[9]; fill_seq_d(A,3,3,3);
    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, 3,3, 1.0, A, 3, 0.0, C, 3);
    return check_d(C, 9);
}
int t_csyrk() {
    float complex alpha=1.0f, beta=0.0f;
    float complex A[9], C[9]; fill_seq_c(A,3,3,3);
    cblas_csyrk(CblasRowMajor, CblasUpper, CblasNoTrans, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}
int t_zsyrk() {
    double complex alpha=1.0, beta=0.0;
    double complex A[9], C[9]; fill_seq_z(A,3,3,3);
    cblas_zsyrk(CblasColMajor, CblasLower, CblasConjTrans, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}

int t_ssyr2k() {
    float A[9], B[9], C[9]; fill_seq_f(A,3,3,3); fill_seq_f(B,3,3,3);
    cblas_ssyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, 3,3, 1.0f, A, 3, B, 3, 0.0f, C, 3);
    return check_f(C, 9);
}
int t_dsyr2k() {
    double A[9], B[9], C[9]; fill_seq_d(A,3,3,3); fill_seq_d(B,3,3,3);
    cblas_dsyr2k(CblasColMajor, CblasLower, CblasTrans, 3,3, 1.0, A, 3, B, 3, 0.0, C, 3);
    return check_d(C, 9);
}
int t_csyr2k() {
    float complex alpha=1.0f, beta=0.0f;
    float complex A[9], B[9], C[9]; fill_seq_c(A,3,3,3); fill_seq_c(B,3,3,3);
    cblas_csyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}
int t_zsyr2k() {
    double complex alpha=1.0, beta=0.0;
    double complex A[9], B[9], C[9]; fill_seq_z(A,3,3,3); fill_seq_z(B,3,3,3);
    cblas_zsyr2k(CblasColMajor, CblasLower, CblasConjTrans, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}

int t_strmm() {
    float A[9], B[9]; init_f(A,9,1.0f); fill_seq_f(B,3,3,3);
    cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 3,3, 1.0f, A, 3, B, 3);
    return check_f(B, 9);
}
int t_dtrmm() {
    double A[9], B[9]; init_d(A,9,1.0); fill_seq_d(B,3,3,3);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit, 3,3, 1.0, A, 3, B, 3);
    return check_d(B, 9);
}
int t_ctrmm() {
    float complex alpha=1.0f;
    float complex A[9], B[9]; init_c(A,9,1.0f); fill_seq_c(B,3,3,3);
    cblas_ctrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3);
    return 1;
}
int t_ztrmm() {
    double complex alpha=1.0;
    double complex A[9], B[9]; init_z(A,9,1.0); fill_seq_z(B,3,3,3);
    cblas_ztrmm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, CblasUnit, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3);
    return 1;
}

int t_strsm() {
    float A[9], B[9]; 
    for(int i=0;i<9;i++) A[i] = (i%4==0)?1.0f:0.0f; 
    fill_seq_f(B,3,3,3);
    cblas_strsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit, 3,3, 1.0f, A, 3, B, 3);
    return check_f(B, 9);
}
int t_dtrsm() {
    double A[9], B[9];
    for(int i=0;i<9;i++) A[i] = (i%4==0)?1.0:0.0;
    fill_seq_d(B,3,3,3);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 3,3, 1.0, A, 3, B, 3);
    return check_d(B, 9);
}
int t_ctrsm() {
    float complex alpha=1.0f;
    float complex A[9], B[9];
    for(int i=0;i<9;i++) A[i] = (i%4==0)?1.0f:0.0f;
    fill_seq_c(B,3,3,3);
    cblas_ctrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3);
    return 1;
}
int t_ztrsm() {
    double complex alpha=1.0;
    double complex A[9], B[9];
    for(int i=0;i<9;i++) A[i] = (i%4==0)?1.0:0.0;
    fill_seq_z(B,3,3,3);
    cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, CblasNonUnit, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3);
    return 1;
}

int t_chemm() {
    float complex alpha=1.0f, beta=0.0f;
    float complex A[9], B[9], C[9]; init_c(A,9,1.0f); fill_seq_c(B,3,3,3); fill_seq_c(C,3,3,3);
    cblas_chemm(CblasRowMajor, CblasLeft, CblasUpper, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}
int t_zhemm() {
    double complex alpha=1.0, beta=0.0;
    double complex A[9], B[9], C[9]; init_z(A,9,1.0); fill_seq_z(B,3,3,3); fill_seq_z(C,3,3,3);
    cblas_zhemm(CblasColMajor, CblasRight, CblasLower, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, CPTR(&beta), CPTR(C), 3);
    return 1;
}


int t_cherk() {
    float alpha=1.0f, beta=0.0f; 
    float complex A[9], C[9]; fill_seq_c(A,3,3,3);
    cblas_cherk(CblasRowMajor, CblasUpper, CblasNoTrans, 3,3, alpha, CPTR(A), 3, beta, CPTR(C), 3);
    return 1;
}
int t_zherk() {
    double alpha=1.0, beta=0.0; 
    double complex A[9], C[9]; fill_seq_z(A,3,3,3);
    cblas_zherk(CblasColMajor, CblasLower, CblasConjTrans, 3,3, alpha, CPTR(A), 3, beta, CPTR(C), 3);
    return 1;
}


int t_cher2k() {
    float complex alpha=1.0f; float beta=0.0f;
    float complex A[9], B[9], C[9]; fill_seq_c(A,3,3,3); fill_seq_c(B,3,3,3);
    cblas_cher2k(CblasRowMajor, CblasUpper, CblasNoTrans, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, beta, CPTR(C), 3);
    return 1;
}
int t_zher2k() {
    double complex alpha=1.0; double beta=0.0;
    double complex A[9], B[9], C[9]; fill_seq_z(A,3,3,3); fill_seq_z(B,3,3,3);
    cblas_zher2k(CblasColMajor, CblasLower, CblasConjTrans, 3,3, CPTR(&alpha), CPTR(A), 3, CPTR(B), 3, beta, CPTR(C), 3);
    return 1;
}

int main() {
    TestSuite ts;
    ts_init(&ts);

    printf(CYAN "=== ВСЕ ТЕСТЫ CBLAS LEVEL 3 ===" RESET "\n\n");

    printf(YELLOW "--- GEMM ---" RESET "\n");
    ts_report(&ts, run_safe(t_sgemm), "cblas_sgemm");
    ts_report(&ts, run_safe(t_dgemm), "cblas_dgemm");
    ts_report(&ts, run_safe(t_cgemm), "cblas_cgemm");
    ts_report(&ts, run_safe(t_zgemm), "cblas_zgemm");

    printf(YELLOW "\n--- SYMM ---" RESET "\n");
    ts_report(&ts, run_safe(t_ssymm), "cblas_ssymm");
    ts_report(&ts, run_safe(t_dsymm), "cblas_dsymm");
    ts_report(&ts, run_safe(t_csymm), "cblas_csymm");
    ts_report(&ts, run_safe(t_zsymm), "cblas_zsymm");

    printf(YELLOW "\n--- SYRK ---" RESET "\n");
    ts_report(&ts, run_safe(t_ssyrk), "cblas_ssyrk");
    ts_report(&ts, run_safe(t_dsyrk), "cblas_dsyrk");
    ts_report(&ts, run_safe(t_csyrk), "cblas_csyrk");
    ts_report(&ts, run_safe(t_zsyrk), "cblas_zsyrk");

    printf(YELLOW "\n--- SYR2K ---" RESET "\n");
    ts_report(&ts, run_safe(t_ssyr2k), "cblas_ssyr2k");
    ts_report(&ts, run_safe(t_dsyr2k), "cblas_dsyr2k");
    ts_report(&ts, run_safe(t_csyr2k), "cblas_csyr2k");
    ts_report(&ts, run_safe(t_zsyr2k), "cblas_zsyr2k");

    printf(YELLOW "\n--- TRMM ---" RESET "\n");
    ts_report(&ts, run_safe(t_strmm), "cblas_strmm");
    ts_report(&ts, run_safe(t_dtrmm), "cblas_dtrmm");
    ts_report(&ts, run_safe(t_ctrmm), "cblas_ctrmm");
    ts_report(&ts, run_safe(t_ztrmm), "cblas_ztrmm");

    printf(YELLOW "\n--- TRSM ---" RESET "\n");
    ts_report(&ts, run_safe(t_strsm), "cblas_strsm");
    ts_report(&ts, run_safe(t_dtrsm), "cblas_dtrsm");
    ts_report(&ts, run_safe(t_ctrsm), "cblas_ctrsm");
    ts_report(&ts, run_safe(t_ztrsm), "cblas_ztrsm");

    printf(YELLOW "\n--- HEMM/HERK/HER2K ---" RESET "\n");
    ts_report(&ts, run_safe(t_chemm), "cblas_chemm");
    ts_report(&ts, run_safe(t_zhemm), "cblas_zhemm");
    ts_report(&ts, run_safe(t_cherk), "cblas_cherk");
    ts_report(&ts, run_safe(t_zherk), "cblas_zherk");
    ts_report(&ts, run_safe(t_cher2k), "cblas_cher2k");
    ts_report(&ts, run_safe(t_zher2k), "cblas_zher2k");

    ts_summary(&ts);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    return (ts.passed == ts.total) ? 0 : 1;
}