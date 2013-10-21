"""
Attempt to implement a CUDA-based SPH renderer
"""

import numba
from numbapro import vectorize, cuda
from numba import autojit, jit, double, int32, void, float32
from numbapro import prange
import numpy as np
import pynbody
import scipy.integrate as integrate
import math
import time
import bisect

@autojit
def get_tile_ids(tiles_physical,xmin,ymin,xs,ys,tileids) :
    t_dx = tiles_physical[0,1]-tiles_physical[0,0]
    t_dy = tiles_physical[0,3]-tiles_physical[0,2]

    x_ind = np.floor((xs-xmin)/t_dx)
    y_ind = np.floor((ys-ymin)/t_dy)
    tileids[:] = x_ind*(np.sqrt(tiles_physical.shape[0]))+y_ind


def cu_template_render_image(s,nx,ny,xmin,xmax, qty='rho',timing = False, nthreads=128):
    """
    CPU part of the SPH render code that executes the rendering on the GPU
    
    does some basic particle set prunning and sets up the image
    tiles. It launches cuda kernels for rendering the individual sections of the image
    """
    import pycuda.driver as drv
    import pycuda.tools
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    global_start = time.clock()

    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]
    ymin,ymax = xmin,xmax

    # ----------------------
    # setup the global image
    # ----------------------
    image = np.zeros((nx,ny),dtype=np.float32)
    
    dx = float32((xmax-xmin)/nx)
    dy = float32((ymax-ymin)/ny)
    
    x_start = xmin+dx/2
    y_start = ymin+dy/2

    zplane = 0.0

    # ------------------------------------
    # trim particles based on image limits
    # ------------------------------------
    start = time.clock()
    ind = np.where(#(np.abs(zs-zplane) < 2*hs) & 
                   (xs > xmin-2*hs) & (xs < xmax+2*hs) & 
                   (ys > ymin-2*hs) & (ys < ymax+2*hs))[0]
    xs,ys,zs,hs,qts,mass,rhos = (xs[ind],ys[ind],zs[ind],hs[ind],qts[ind],mass[ind],rhos[ind])
    if timing: print '<<< Initial particle selection took %f s'%(time.clock()-start)

    # set the render quantity 
    qts *= mass/rhos

    #
    # bin particles by how many pixels they need in their kernel
    #
    start = time.clock()
    npix = 2.0*hs/dx
    dbin = np.digitize(npix,np.arange(1,npix.max()))
    dbin_sortind = dbin.argsort()
    dbin_sorted = dbin[dbin_sortind]
    xs,ys,zs,hs,qts = (xs[dbin_sortind],ys[dbin_sortind],zs[dbin_sortind],hs[dbin_sortind],qts[dbin_sortind])
    if timing: print '<<< Bin sort done in %f'%(time.clock()-start)
    
    # -----------------------------------------------------------------
    # set up the image slices -- max. size is 100x100 pixels 
    # in this step only process particles that need kernels < 40 pixels
    # tiles are 100x100 = 1e4 pixels x 4 bytes = 40k
    # kernels are 31x31 pixels max = 3844 bytes
    # max shared memory size is 48k
    # -----------------------------------------------------------------
    
    start = time.clock()
    tiles_pix, tiles_physical = make_tiles(nx,ny,xmin,xmax,ymin,ymax,100)
    if timing: print '<<< Tiles made in %f s'%(time.clock()-start)

    start = time.clock()
    ind = np.searchsorted(hs,[0.0,15.*dx/2.0]) # indices of particles with 2h/dx < 30 pixels
    if timing: print '<<< Sorted search took %f s'%(time.clock()-start)

        
    # -----------------------------------------
    # set up streams and transfer particle data
    # -----------------------------------------
    
    start = time.clock()

    xs_s, ys_s, qts_s, hs_s = [xs[:ind[1]], ys[:ind[1]], qts[:ind[1]], hs[:ind[1]]]

    xs_gpu = drv.mem_alloc(xs_s.astype(np.float32).nbytes)
    drv.memcpy_htod(xs_gpu,xs_s.astype(np.float32))
    
    ys_gpu = drv.mem_alloc(ys_s.astype(np.float32).nbytes)
    drv.memcpy_htod(ys_gpu,ys_s.astype(np.float32))
    
    qts_gpu = drv.mem_alloc(qts_s.astype(np.float32).nbytes)
    drv.memcpy_htod(qts_gpu,qts_s.astype(np.float32))
    
    hs_gpu = drv.mem_alloc(hs_s.astype(np.float32).nbytes)
    drv.memcpy_htod(hs_gpu,hs_s.astype(np.float32))
    
    im_gpu = drv.mem_alloc(image.astype(np.float32).nbytes)
    drv.memcpy_htod(im_gpu,image.astype(np.float32))

    inds_gpu = drv.mem_alloc(xs.astype(np.int32).nbytes)
    
    if timing: print "<<< Data transfer to device took %f s"%(time.clock()-start)

    code = file('/home/itp/roskar/homegrown/template_kernel.cu').read()
    
    mod = SourceModule(code)

    kernel = mod.get_function("tile_render_kernel")
   
    Ntiles = tiles_pix.shape[0]
        
    streams = [drv.Stream() for i in range(16)]    
    
    tile_start = time.clock()

    drv.start_profiler()
    for i in xrange(Ntiles) :
        
        tile   = tiles_pix[i]
        tile_p = tiles_physical[i]
    
        xmin_t, xmax_t, ymin_t, ymax_t = tile
        xmin_p, xmax_p, ymin_p, ymax_p  = tile_p
    
        nx_tile = xmax_t-xmin_t+1
        ny_tile = ymax_t-ymin_t+1
        where_start = time.clock()
        inds = np.where((xs_s + 2*hs_s >= xmin_p) & (xs_s - 2*hs_s <= xmax_p) & 
                        (ys_s + 2*hs_s >= ymin_p) & (ys_s - 2*hs_s <= ymax_p))[0]                     
        if timing: print '<<< Tile where took %f s'%(time.clock()-where_start)
        if inds.shape[0] > 0 : 
            start = time.clock()

            my_stream = streams[i%16]

            kmax = int(math.ceil(hs_s[inds].max()*2.0/dx*2.0))+1
            kmin = int(math.floor(hs_s[inds].min()*2.0/dx*2.0))
            
            # make everything the right size
            kmax,kmin,xmin_t,xmax_t,ymin_t,ymax_t = map(np.int32,[kmax,kmin,xmin_t,xmax_t,ymin_t,ymax_t])
            xmin_p,xmax_p,ymin_p,ymax_p = map(np.float32, [xmin_p,xmax_p,ymin_p,ymax_p])
            
            drv.memcpy_htod_async(inds_gpu,inds.astype(np.int32),stream=my_stream)

            kernel(xs_gpu,ys_gpu,qts_gpu,hs_gpu,inds_gpu,np.int32(len(inds)),
                   kmin,kmax,xmin_p,xmax_p,ymin_p,ymax_p,xmin_t,xmax_t,ymin_t,ymax_t,
                   im_gpu,np.int32(image.shape[0]),np.int32(image.shape[1]),
                   block=(nthreads,1,1),stream=my_stream)

    if timing: print '<<< %d kernels launched in %f s'%(Ntiles,time.clock()-tile_start)
        # close if inds.shape[0]>0
    # close tile for loop
    

    # ----------------------------------------------------------------------------------
    # process the particles with large smoothing lengths concurrently with GPU execution
    # ----------------------------------------------------------------------------------
    if ind[1] != len(xs) : 
        start = time.clock()
        image2 = (template_kernel_cpu(xs[ind[1]:],ys[ind[1]:],qts[ind[1]:],hs[ind[1]:],
                                      nx,ny,xmin,xmax,ymin,ymax)).T
        if timing: print '<<< Processing %d particles with large smoothing lengths took %e s'%(len(xs)-ind[1],
                                                                                               time.clock()-start)
    drv.Context.synchronize()
    drv.memcpy_dtoh(image,im_gpu)
    drv.stop_profiler()
    if timing: print '<<< %d tiles rendered in %f s'%(Ntiles,time.clock()-tile_start)

    if timing: print '<<< Total render done in %f s\n'%(time.clock()-global_start)
    
    return image


@autojit
def make_tiles(nx, ny, x_phys_min, x_phys_max, y_phys_min, y_phys_max, max_dim) : 
    # size of pixels in physical space
    dx = float(x_phys_max-x_phys_min)/float(nx)
    dy = float(y_phys_max-y_phys_min)/float(ny)
    
    nx_tiles = np.ceil(float(nx)/float(max_dim))
    ny_tiles = np.ceil(float(ny)/float(max_dim))
    n_tiles = nx_tiles*ny_tiles

    limits = np.zeros((n_tiles,4),dtype=np.int32)
    limits_physical = np.zeros((n_tiles,4),dtype=np.float)

    for i in range(int(nx_tiles)) : 
        xmin = i*max_dim
        xmax = (i+1)*max_dim -1 if i < nx_tiles-1 else nx-1
        
        for j in range(int(ny_tiles)) :
            ymin = j*max_dim
            ymax = (j+1)*max_dim -1 if j < ny_tiles-1 else ny-1
            
            limits[i*nx_tiles + j] = [xmin,xmax,ymin,ymax]

            limits_physical[i*nx_tiles+j] = [x_phys_min+dx*xmin,x_phys_min+(xmax+1)*dx,
                                             y_phys_min+dy*ymin,y_phys_min+(ymax+1)*dy]
                                             
                                             
    
    

    return limits, limits_physical

def test_template_render(s,nx,ny,xmin,xmax,qty='rho',sort_arr = None, timing = False) : 
    start = time.clock()
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]
    res = cu_template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax,sort_arr=sort_arr, timing=timing)
    print '<<< Done after %f seconds'%(time.clock()-start)
    return res 


