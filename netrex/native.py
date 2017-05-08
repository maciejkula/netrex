import glob
import os

from cffi import FFI

import numpy as np


class Extension:

    def __init__(self, lib):

        self._lib = lib
        self._ffi = FFI()

    def _cast(self, x):

        return self._ffi.cast('float *', x.ctypes.data)

    def predict_float_256(self,
                          user_vector,
                          item_vectors,
                          user_bias,
                          item_biases,
                          out=None):

        cast = self._cast

        if out is None:
            out = np.zeros_like(item_biases)

        num_items, latent_dim = item_vectors.shape

        self._lib.predict_float_256(
            cast(user_vector),
            cast(item_vectors),
            user_bias,
            cast(item_biases),
            cast(out),
            num_items,
            latent_dim)

        return out

    def predict_xnor_256(self,
                         user_vector,
                         item_vectors,
                         user_bias,
                         item_biases,
                         user_norm,
                         item_norms,
                         out=None):

        cast = self._cast

        if out is None:
            out = np.zeros_like(item_biases)

        num_items, latent_dim = item_vectors.shape

        # Express latent dimension in term of floats
        latent_dim = latent_dim // (4 // item_vectors.itemsize)

        self._lib.predict_xnor_256(
            cast(user_vector),
            cast(item_vectors),
            user_bias,
            cast(item_biases),
            user_norm,
            cast(item_norms),
            cast(out),
            num_items,
            latent_dim)

        return out


def _build_module():

    ffibuilder = FFI()
    ffibuilder.set_source("_native", None)
    ffibuilder.cdef("""
    void predict_float_256(float* user_vector,
                       float* item_vectors,
                       float user_bias,
                       float* item_biases,
                       float* out,
                       intptr_t num_items,
                       intptr_t latent_dim);
    void predict_xnor_256(float* user_vector,
                      float* item_vectors,
                      float user_bias,
                      float* item_biases,
                      float user_norm,
                      float* item_norm,
                      float* out,
                      intptr_t num_items,
                      intptr_t latent_dim);
    """)

    ffibuilder.compile(verbose=False)


def get_lib():

    from netrex._native import ffi

    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'libpredict*.so')

    libs = glob.glob(path)

    if not libs:
        raise Exception('Compiled extension not found under {}'.format(path))

    return Extension(ffi.dlopen(libs[0]))
