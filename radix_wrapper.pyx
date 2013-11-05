import numpy as np
cimport numpy as np

ctypedef np.int32_t int_t

cdef extern from "radix_sort.h":
    struct Particle:
        float x
        float y
        float qt
        float h

    float _radix_sort "radix_sort"(int_t *keys, Particle *ps, int offset, int num_items)
    
    
def radix_sort(np.ndarray[int_t,ndim=1] keys, np.ndarray[Particle,ndim=1] ps, int offset, int num_items):
    return _radix_sort(<int_t*> keys.data, <Particle*> ps.data, offset, num_items)
