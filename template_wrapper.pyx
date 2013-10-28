import numpy as np
cimport numpy as np

#DTYPE = np.float32
#ctypedef np.float32_t DTYPE_t
ctypedef np.float32_t float_t
ctypedef np.double_t double_t
ctypedef np.int32_t int_t

cdef extern from "template_kernel.c":
     void _kernel_func "kernel_func"(double_t *kernel, double h, double max_d, int ksize)	
     
     void _kernel_distance "kernel_distance"(double_t *kernel, double dx, double dy, int ksize)
     
     void _tile_render_kernel "tile_render_kernel"(double_t *xs, double_t *ys, 
                                                   double_t *qts, double_t *hs, 
                                                   int Npart,  
                                                   int xmin, int xmax, 
                                                   int ymin, int ymax, 
                                                   double_t *image, int nx, int ny)


def kernel_func(np.ndarray[double_t,ndim=1] kernel, double h, double max_d, int ksize) :
    _kernel_func(<double_t*> kernel.data,h,max_d,ksize)

def kernel_distance(np.ndarray[double_t,ndim=1] kernel, double dx, double dy, int ksize) :
    _kernel_distance(<double_t*> kernel.data,dx,dy,ksize)
     
def tile_render_kernel(xs, ys, 
                       qts, hs, 
                       int Npart, int xmin, int xmax, int ymin, int ymax, 
                       image, int nx, int ny) :

    cdef np.ndarray[np.double_t,ndim=1,mode="c"] xs_c
    xs_c = np.ascontiguousarray(xs,dtype=np.double)

    cdef np.ndarray[np.double_t,ndim=1,mode="c"] ys_c
    ys_c = np.ascontiguousarray(ys,dtype=np.double)

    cdef np.ndarray[np.double_t,ndim=1,mode="c"] qts_c
    qts_c = np.ascontiguousarray(qts,dtype=np.double)

    cdef np.ndarray[np.double_t,ndim=1,mode="c"] hs_c
    hs_c = np.ascontiguousarray(hs,dtype=np.double)

    cdef np.ndarray[np.double_t,ndim=1,mode="c"] image_c
    image_c = np.ascontiguousarray(image,dtype=np.double)

    _tile_render_kernel(&xs_c[0], &ys_c[0], &qts_c[0], &hs_c[0], Npart, xmin, xmax, ymin, ymax, &image_c[0], nx, ny)
