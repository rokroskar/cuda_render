import numpy as np
cimport numpy as np

ctypedef np.int32_t int_t

cdef extern from "radix_sort.h":
    struct Particle:
        float x
        float y
        float qt
        float h

    float _radix_sort "radix_sort"(int *keys, Particle *ps, int offset, int num_items)
    
    
def radix_sort(long keys, long ps, int offset, int num_items):
    return _radix_sort(<int*> keys, <Particle*> ps, offset, num_items)
