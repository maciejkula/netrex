#include <stdint.h>
#include <x86intrin.h>
#include "libpopcnt.h"


/* void predict_float_256(float* user_vector, */
/*                        float* item_vectors, */
/*                        float user_bias, */
/*                        float* item_biases, */
/*                        float* out, */
/*                        intptr_t num_items, */
/*                        intptr_t latent_dim) { */

/*     float* item_vector; */

/*     __m256 x, y, prediction; */
/*     float scalar_prediction; */
/*     float unpacked[8]; */

/*     for (int i = 0; i < num_items; i++) { */

/*         item_vector = item_vectors + (i * latent_dim); */
/*         prediction = _mm256_xor_ps(prediction, prediction); */

/*         for (int j = 0; j < latent_dim; j += 8) { */
/*             x = _mm256_loadu_ps(item_vector + j); */
/*             y = _mm256_loadu_ps(user_vector + j); */

/*             prediction = _mm256_add_ps( */
/*                 prediction, */
/*                 _mm256_mul_ps(x, y)); */
/*         } */

/*         _mm256_storeu_ps(unpacked, prediction); */

/*         scalar_prediction = item_biases[i] + user_bias; */

/*         for (int j = 0; j < 8; j++) { */
/*             scalar_prediction += unpacked[j]; */
/*         } */

/*         out[i] = scalar_prediction; */
/*     } */
/* } */


void predict_float_256(float* user_vector,
                       float* item_vectors,
                       float user_bias,
                       float* item_biases,
                       float* out,
                       intptr_t num_items,
                       intptr_t latent_dim) {

    float* item_vector;

    float x, y;
    float scalar_prediction;

    for (int i = 0; i < num_items; i++) {

        item_vector = item_vectors + (i * latent_dim);
        scalar_prediction = 0;

        for (int j = 0; j < latent_dim; j++) {
            x = item_vector[j];
            y = user_vector[j];

            scalar_prediction += x * y;
        }

        scalar_prediction += item_biases[i] + user_bias;

        out[i] = scalar_prediction;
    }
}


/* void predict_xnor_256(float* user_vector, */
/*                       float* item_vectors, */
/*                       float user_bias, */
/*                       float* item_biases, */
/*                       float user_norm, */
/*                       float* item_norms, */
/*                       float* out, */
/*                       intptr_t num_items, */
/*                       intptr_t latent_dim) { */

/*     float* item_vector; */

/*     __m256 x, y; */
/*     float scalar_prediction; */
/*     float on_bits; */

/*     float* bits = malloc(latent_dim * sizeof(float)); */

/*     float max_on_bits = latent_dim * 32; */

/*     for (int i = 0; i < latent_dim; i++) { */
/*         bits[i] = 0.0; */
/*     } */

/*     __m256 allbits = _mm256_cmp_ps(_mm256_loadu_ps(bits), */
/*                                    _mm256_loadu_ps(bits), */
/*                                    0); */

/*     for (int i = 0; i < num_items; i++) { */

/*         item_vector = item_vectors + (i * latent_dim); */
/*         scalar_prediction = 0; */

/*         for (int j = 0; j < latent_dim; j += 8) { */
/*             x = _mm256_loadu_ps(item_vector + j); */
/*             y = _mm256_loadu_ps(user_vector + j); */

/*             // XNOR */
/*             _mm256_storeu_ps(bits + j, _mm256_xor_ps(_mm256_xor_ps(x, y), allbits)); */
/*             // _mm256_storeu_ps(bits + j, allbits); */
/*         } */

/*         //on_bits = popcnt_avx2((const __m256i*) bits, latent_dim / 8); */
/*         on_bits = popcnt((const void*) bits, latent_dim * sizeof(float)); */
/*         scalar_prediction = (on_bits - (max_on_bits - on_bits)) * user_norm * item_norms[i]; */

/*         out[i] = scalar_prediction + user_bias + item_biases[i]; */
/*     } */

/*     free(bits); */
/* } */


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
            on_bits += popcnt((const void*) bits, 8 * sizeof(float));
        }

        // Scaling
        scalar_prediction = (on_bits - (max_on_bits - on_bits)) * user_norm * item_norms[i];

        // Biases
        out[i] = scalar_prediction + user_bias + item_biases[i];
    }
}



