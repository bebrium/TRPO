#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int fmax(int *vec, int n, int max) {

    int i;

    if (!vec || (n <=0) ) {
        return -1;
    }
    max = vec[0];
    for (i=1; i <= n; i++) {
        if (max < vec[i]) {
            max = vec[i];
        }
    }
    return 0;
}

/***************************************/
int main(int argc, char *argv[]) {
	printf("program start");

    int arr[5] = { 17, 21, 44, 2, 60 };

    int max = arr[0];

    if ( fmax(arr, 5, max) != 0 ) {
        printf("strange error\n");
        exit(1);
    }
    printf("max value in the array is %d\n", max);

    return 0;
}