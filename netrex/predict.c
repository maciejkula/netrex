#include <stdint.h>
#include <x86intrin.h>
#include "libpopcnt.h"


int _get_cpuid() {
#if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
#else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1)
    {
        cpuid = get_cpuid();
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
    }
#endif
    return cpuid;
}


/*
 * Count the number of 1 bits in the data array
 * @data: An array
 * @size: Size of data in bytes
 * @cpuid: Result of the cpuid call
 */
static inline uint64_t popcnt_no_cpuid(const void* data, uint64_t size, int cpuid) {
  const uint8_t* ptr = (const uint8_t*) data;
  uint64_t cnt = 0;
  uint64_t i;

#if defined(HAVE_AVX2)

  /* AVX2 requires arrays >= 512 bytes */
  if ((cpuid & bit_AVX2) &&
      size >= 512)
  {
    align_avx2(&ptr, &size, &cnt);
    cnt += popcnt_avx2((const __m256i*) ptr, size / 32);
    ptr += size - size % 32;
    size = size % 32;
  }

#endif

#if defined(HAVE_POPCNT)

  if (cpuid & bit_POPCNT)
  {
    cnt += popcnt64_unrolled((const uint64_t*) ptr, size / 8);
    ptr += size - size % 8;
    size = size % 8;
    for (i = 0; i < size; i++)
      cnt += popcnt64(ptr[i]);

    return cnt;
  }

#endif

  /* pure integer popcount algorithm */
  for (i = 0; i < size; i++)
    cnt += popcount64(ptr[i]);

  return cnt;
}


void predict_float_256(float* user_vector,
                       float* item_vectors,
                       float user_bias,
                       float* item_biases,
                       float* out,
                       intptr_t num_items,
                       intptr_t latent_dim) {

    float* item_vector;

    __m256 x, y, product, prediction;
    float scalar_prediction;
    float unpacked[8];

    for (int i = 0; i < num_items; i++) {

        item_vector = item_vectors + (i * latent_dim);
        prediction = _mm256_xor_ps(prediction, prediction);

        for (int j = 0; j < latent_dim; j += 8) {
            x = _mm256_loadu_ps(item_vector + j);
            y = _mm256_loadu_ps(user_vector + j);

            product = _mm256_mul_ps(x, y);

            prediction = _mm256_add_ps(
                prediction,
                product);
        }

        _mm256_storeu_ps(unpacked, prediction);

        scalar_prediction = item_biases[i] + user_bias;

        for (int j = 0; j < 8; j++) {
            scalar_prediction += unpacked[j];
        }

        out[i] = scalar_prediction;
    }
}


void predict_xnor_256(float* user_vector,
                      float* item_vectors,
                      float user_bias,
                      float* item_biases,
                      float user_norm,
                      float* item_norms,
                      float* out,
                      intptr_t num_items,
                      intptr_t latent_dim) {

    float* item_vector;

    __m256 x, y;
    float scalar_prediction;
    float on_bits;

    int cpuid = _get_cpuid();

    float bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    float max_on_bits = latent_dim * 32;

    __m256 allbits = _mm256_cmp_ps(_mm256_loadu_ps(bits),
                                   _mm256_loadu_ps(bits),
                                   0);

    for (int i = 0; i < num_items; i++) {

        item_vector = item_vectors + (i * latent_dim);
        scalar_prediction = 0;
        on_bits = 0;

        for (int j = 0; j < latent_dim; j += 8) {
            x = _mm256_loadu_ps(item_vector + j);
            y = _mm256_loadu_ps(user_vector + j);

            // XNOR
            _mm256_storeu_ps(bits, _mm256_xor_ps(_mm256_xor_ps(x, y), allbits));

            // Bitcount
            on_bits += popcnt_no_cpuid((const void*) bits, 8 * sizeof(float), cpuid);
            // on_bits += popcnt((const void*) bits, 8 * sizeof(float));
        }

        // Scaling
        scalar_prediction = (on_bits - (max_on_bits - on_bits)) * user_norm * item_norms[i];

        // Biases
        out[i] = scalar_prediction + user_bias + item_biases[i];
    }
}
