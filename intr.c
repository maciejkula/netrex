#include <x86intrin.h>
#include <stdio.h>

int main(void) {

    int res = 0;
    int c[8];

    int a[8] = {1,2,3,4,5,6,7,8};
    int b[8] = {1,2,-4,-5,-6,-7,-8,-10};

    __m256 allbits = _mm256_and_ps(
        _mm256_loadu_ps((float*) a),
        _mm256_loadu_ps((float*) a)
        );

    for (int i=0;i < 100000000;i++) {

        __m256 a8 = _mm256_loadu_ps((float*)a);
        __m256 b8 = _mm256_loadu_ps((float*)b);
        __m256 xor = _mm256_xor_ps(a8,b8);
        __m256 c8 = _mm256_xor_ps(allbits, xor);

        _mm256_storeu_ps((float*)c, c8);

        res += c8[0];
    }

    printf("%i", res);
}


/* int main(void) { */

/*     int res = 0; */

/*     int a[8] = {1,2,3,4,5,6,7,8}; */
/*     int b[8] = {1,2,-4,-5,-6,-7,-8,-10}; */

/*     for (int i=0;i < 100000000;i++) { */

/*         for (int j=0; j<8; j++) { */
/*             res += a[j] & b[j]; */
/*         } */
/*     } */

/*     printf("%i", res); */
/* } */
