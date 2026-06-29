#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *calc(void *arg);

static unsigned long long res;
static unsigned long long N_loops = 1000000;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char **argv) {

    pthread_t *p_th;
    int ret, i, num_ths, *th_id_args;

    if (argc != 2) {
        printf("usage: ./a.out num_threads\n");
        exit(1);
    }
    num_ths = atoi(argv[1]);
    
    if ((num_ths < 1) || (num_ths > 24)) { num_ths = 6; }

    p_th = (pthread_t *)malloc(sizeof(pthread_t)*num_ths);
    if (!p_th) {
        perror("malloc pthread_t array");
        exit(1);
    }
    if (!(th_id_args = (int *)malloc(sizeof(int)*num_ths))) {
        perror("malloc th_id_args array");
        exit(1);
    }

    res = 0;
    for (i = 0; i < num_ths; i++) {
        th_id_args[i] = i;
        ret = pthread_create(&p_th[i], 0, calc, &th_id_args[i]);
        if (ret) {
            perror("Error pthread_create\n");
            exit(1);
        }
    }

    for (i = 0; i < num_ths; i++) {
        pthread_join(p_th[i], 0);
    }
    printf("res = %llu\n", res);

    free ((int *)th_id_args);
    free ((pthread_t *)p_th);
    pthread_mutex_destroy(&mutex);
    exit(0);
}

void *calc(void *arg) {

    int myid, i;
    unsigned long long local_sum = 0;
    myid = *((int *)arg);

    printf("hello I'm thread %d with pthread_id %lu\n",
            myid, pthread_self());

    for (i = 0; i < N_loops; i++) {
        local_sum += i;
    }

    pthread_mutex_lock(&mutex);
    res += local_sum;
    pthread_mutex_unlock(&mutex);

    return (void *)0;
}