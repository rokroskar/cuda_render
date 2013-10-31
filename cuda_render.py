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
from bisect import bisect_left, bisect_right

@autojit(nopyton=True)
def get_tile_ids(tiles_physical,xmin,ymin,ps,tileids) :
    t_dx = tiles_physical[0,1]-tiles_physical[0,0]
    t_dy = tiles_physical[0,3]-tiles_physical[0,2]

    x_ind = np.floor((ps['x']-xmin)/t_dx)
    y_ind = np.floor((ps['y']-ymin)/t_dy)
    
    tileids[:] = x_ind*np.sqrt(tiles_physical.shape[0]) + y_ind


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
            
            limits[i*ny_tiles + j] = [xmin,xmax,ymin,ymax]

            limits_physical[i*nx_tiles+j] = [x_phys_min+dx*xmin,x_phys_min+(xmax+1)*dx,
                                             y_phys_min+dy*ymin,y_phys_min+(ymax+1)*dy]
                                             
    return limits, limits_physical

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

    

    start = time.clock()
    # construct an array of particles
    ps = np.empty(len(s),dtype=[('x','f4'),('y','f4'),('z','f4'),('qt','f4'),('h','f4')])
    
    with s.immediate_mode : 
        ps['x'],ps['y'],ps['z'],ps['qt'],ps['h'] = [s[arr] for arr in ['x','y','z','mass','smooth']]

    if timing: print '<<< Forming particle struct took %f s'%(time.clock()-start)

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

    # ------------------------------------------------------------------------------------------------
    # trim particles based on smoothing length -- the GPU will only render those that need < 32 pixels
    # ------------------------------------------------------------------------------------------------

    start = time.clock()
    gpu_bool = 2*ps['h'] < 15.*dx
    
    ps_gpu = ps[gpu_bool]
    ps_cpu = ps[~gpu_bool]
    del(ps)
    if timing: '<<< Setting up gpu/cpu particle struct arrays took %f s'%(time.clock()-start)

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

    Ntiles = tiles_pix.shape[0]

    # -------------------------------------------------------------
    # set up streams and figure out particle distributions per tile 
    # -------------------------------------------------------------
    
    streams = [drv.Stream() for i in range(16)]    
    
    # -------------------------------
    # set up tile distribution kernel
    # -------------------------------
    code = file('/home/itp/roskar/homegrown/template_kernel.cu').read()
    mod = SourceModule(code)
    tile_histogram = mod.get_function("tile_histogram")
   
    # allocate histogram array
    hist = np.zeros(Ntiles,dtype=np.int32)
    
    # transfer histogram array and particle data to GPU
    hist_gpu = drv.mem_alloc(hist.nbytes)
    drv.memcpy_htod(hist_gpu,hist)
    
    ps_on_gpu = drv.mem_alloc(ps_gpu.nbytes)
    drv.memcpy_htod(ps_on_gpu,ps_gpu)

    # make everything the right size
    xmin,xmax,ymin,ymax = map(np.float32, [xmin,xmax,ymin,ymax])
    nx,ny,Ntiles = map(np.int32, [nx,ny,Ntiles])
    
    print ps_gpu['x'].min(), ps_gpu['y'].min()

    tile_histogram(ps_on_gpu,hist_gpu,np.int32(len(ps_gpu)),xmin,xmax,ymin,ymax,nx,ny,Ntiles,block=(512,1,1))
    
    drv.Context.synchronize()
    drv.memcpy_dtoh(hist,hist_gpu)

    return hist
    
    start = time.clock()
    tile_ids = np.empty(len(ps_gpu),dtype=np.int32) 
    get_tile_ids(tiles_physical,xmin,ymin,ps_gpu,tile_ids)
    if timing: print '<<< Generating tile IDs took %f s'%(time.clock()-start)

    
    start = time.clock()
    # sort particles by tile ID
    inds = tile_ids.argsort()
    tile_ids = tile_ids[inds]
    for n in ps_gpu.dtype.names : ps_gpu[n] = ps_gpu[n][inds]
    if timing: print '<<< Sorting particle struct took %f s'%(time.clock() - start)

    start = time.clock()
    # set up tile slices 
    slices = []
    n_per_tile = []
    prev = bisect_left(tile_ids,0)
    for i in range(0,Ntiles) : 
        next = bisect_left(tile_ids,i+1,prev)
        slices.append(slice(prev,next))
        n_per_tile.append(next-prev)
        prev = next
    if timing: print '<<< Setting up tile slices took %f s'%(time.clock()-start)
  
    # ----------------------------------------
    # allocate memory on the GPU for each tile
    # ----------------------------------------
    
    xs_gpu = []
    ys_gpu = []
    qt_gpu = []
    hs_gpu = []

    for i in range(Ntiles) : 
        if n_per_tile[i] > 0 : 
            xs_gpu.append(drv.mem_alloc(n_per_tile[i]*4))
            ys_gpu.append(drv.mem_alloc(n_per_tile[i]*4))
            qt_gpu.append(drv.mem_alloc(n_per_tile[i]*4))
            hs_gpu.append(drv.mem_alloc(n_per_tile[i]*4))
        else : 
            xs_gpu.append(None)
            ys_gpu.append(None)
            qt_gpu.append(None)
            hs_gpu.append(None)
    
    im_gpu = drv.mem_alloc(image.astype(np.float32).nbytes)
    drv.memcpy_htod(im_gpu,image.astype(np.float32))

    # ----------------------
    # set up the kernel code
    # ----------------------
    code = file('/home/itp/roskar/homegrown/template_kernel.cu').read()
    mod = SourceModule(code)
    kernel = mod.get_function("tile_render_kernel")
   

    # ---------------------------------------------------------------------
    # start going through tiles -- first send data to GPU then start kernel
    # ---------------------------------------------------------------------
    
        
    
    tile_start = time.clock()

    drv.start_profiler()
    for i in xrange(Ntiles) :
        sl = slices[i]
        
        if n_per_tile[i] > 0 : 
            my_stream = streams[i%16]
        
            x,y,qt,h = [ps_gpu[arr][sl] for arr in ['x','y','qt','h']]
        
            npix = 2.0*h/dx
            dbin = np.digitize(npix,np.arange(1,npix.max()))
            sortind = dbin.argsort()
        
            x,y,qt,h = x[sortind],y[sortind],qt[sortind],h[sortind]

            drv.memcpy_htod_async(xs_gpu[i],ps_gpu['x'][sl][sortind].astype(np.float32), stream = my_stream)
            drv.memcpy_htod_async(ys_gpu[i],ps_gpu['y'][sl][sortind].astype(np.float32), stream = my_stream)
            drv.memcpy_htod_async(qt_gpu[i],ps_gpu['qt'][sl][sortind].astype(np.float32), stream = my_stream)
            drv.memcpy_htod_async(hs_gpu[i],ps_gpu['h'][sl][sortind].astype(np.float32), stream = my_stream)
    
            tile   = tiles_pix[i]
            tile_p = tiles_physical[i]
    
            xmin_t, xmax_t, ymin_t, ymax_t = tile
            xmin_p, xmax_p, ymin_p, ymax_p  = tile_p
    
            nx_tile = xmax_t-xmin_t+1
            ny_tile = ymax_t-ymin_t+1
        
            start = time.clock()
        
            # make everything the right size
            xmin_t,xmax_t,ymin_t,ymax_t = map(np.int32,[xmin_t,xmax_t,ymin_t,ymax_t])
            xmin_p,xmax_p,ymin_p,ymax_p = map(np.float32, [xmin_p,xmax_p,ymin_p,ymax_p])
            
            kernel(xs_gpu[i],ys_gpu[i],qt_gpu[i],hs_gpu[i],np.int32(n_per_tile[i]),
                   xmin_p,xmax_p,ymin_p,ymax_p,xmin_t,xmax_t,ymin_t,ymax_t,
                   im_gpu,np.int32(image.shape[0]),np.int32(image.shape[1]),
                   block=(nthreads,1,1),stream=my_stream)

    if timing: print '<<< %d kernels launched in %f s'%(Ntiles,time.clock()-tile_start)
    
    # ----------------------------------------------------------------------------------
    # process the particles with large smoothing lengths concurrently with GPU execution
    # ----------------------------------------------------------------------------------
    #if ind[1] != len(xs) : 
    #    start = time.clock()
    #    image2 = (template_kernel_cpu(xs[ind[1]:],ys[ind[1]:],qts[ind[1]:],hs[ind[1]:],
    #                                  nx,ny,xmin,xmax,ymin,ymax)).T
    #    if timing: print '<<< Processing %d particles with large smoothing lengths took %e s'%(len(xs)-ind[1],
    #                                                                                           time.clock()-start)
    drv.Context.synchronize()
    drv.memcpy_dtoh(image,im_gpu)
    drv.stop_profiler()
    if timing: print '<<< %d tiles rendered in %f s'%(Ntiles,time.clock()-tile_start)

    if timing: print '<<< Total render done in %f s\n'%(time.clock()-global_start)
    
    return image



def test_template_render(s,nx,ny,xmin,xmax,qty='rho',sort_arr = None, timing = False) : 
    start = time.clock()
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]
    res = cu_template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax,sort_arr=sort_arr, timing=timing)
    print '<<< Done after %f seconds'%(time.clock()-start)
    return res 


