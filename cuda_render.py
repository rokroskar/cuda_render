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
    from radix_sort import radix_sort

    global_start = time.clock()

    

    start = time.clock()
    # construct an array of particles
    Partstruct = [('x','f4'),('y','f4'),('qt','f4'),('h','f4')]
    ps = drv.pagelocked_empty(len(s),dtype=Partstruct)
    
    with s.immediate_mode : 
        ps['x'],ps['y'],ps['qt'],ps['h'] = [s[arr] for arr in ['x','y','mass','smooth']]

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
  #  gpu_bool = 2*ps['h'] < 15.*dx
    
    ps_gpu = ps#[gpu_bool]
   # ps_cpu = ps[~gpu_bool]
    #del(ps)
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

     
    streams = [drv.Stream() for i in range(16)]    
    
    # ------------------
    # set up the kernels
    # ------------------
    code = file('/home/itp/roskar/homegrown/template_kernel.cu').read()
    mod = SourceModule(code)
    tile_histogram = mod.get_function("tile_histogram")
    distribute_particles = mod.get_function("distribute_particles")
    tile_render_kernel = mod.get_function("tile_render_kernel")
    calculate_keys = mod.get_function("calculate_keys")


    # -------------------------------------------------------------
    # set up streams and figure out particle distributions per tile 
    # -------------------------------------------------------------
   

    # allocate histogram array
    hist = np.zeros(Ntiles,dtype=np.int32)
    
    # transfer histogram array and particle data to GPU
    hist_gpu = drv.mem_alloc(hist.nbytes)
    drv.memcpy_htod(hist_gpu,hist)
    
    start_g = drv.Event()
    end_g = drv.Event()

    start_g.record()
    ps_on_gpu = drv.mem_alloc(ps_gpu.nbytes)
    drv.memcpy_htod(ps_on_gpu,ps_gpu)
    end_g.record()
    end_g.synchronize()

    if timing: print '<<< Particle copy onto GPU took %f ms'%(start_g.time_till(end_g))

    # make everything the right size
    xmin,xmax,ymin,ymax = map(np.float32, [xmin,xmax,ymin,ymax])
    nx,ny,Ntiles = map(np.int32, [nx,ny,Ntiles])

    start_g.record()
    tile_histogram(ps_on_gpu,hist_gpu,np.int32(len(ps_gpu)),xmin,xmax,ymin,ymax,nx,ny,Ntiles,
                   block=(nthreads,1,1),grid=(32,1,1))

    drv.Context.synchronize()
    drv.memcpy_dtoh(hist,hist_gpu)
    end_g.record()
    end_g.synchronize()
    if timing: print '<<< Tile histogram took %f ms'%(start_g.time_till(end_g))
    print "<<< Total particle array = %d"%(hist.sum())

    # ---------------------------------------------------------------------------------
    # figured out the numbers of particles per tile -- set up the tile particle buffers
    # ---------------------------------------------------------------------------------
    ps_tiles = np.empty(hist.sum(),dtype=Partstruct)
    ps_tiles_gpu = drv.mem_alloc(ps_tiles.nbytes)

    tile_offsets = np.array([0],dtype=np.int32)
    tile_offsets = np.append(tile_offsets, hist.cumsum().astype(np.int32))
    tile_offsets_gpu = drv.mem_alloc(tile_offsets.nbytes)
    drv.memcpy_htod(tile_offsets_gpu,tile_offsets)

    start_g.record()
    distribute_particles(ps_on_gpu, ps_tiles_gpu, tile_offsets_gpu, np.int32(len(ps_gpu)), 
                         xmin, xmax, ymin, ymax, nx, ny, Ntiles, 
                         block=(nthreads,1,1), grid=(np.int(Ntiles),1,1), shared=(nthreads*2+1)*4)
    end_g.record()
    end_g.synchronize()
    if timing: print '<<< Particle reshuffling took %f ms'%(start_g.time_till(end_g))
    drv.memcpy_dtoh(ps_tiles, ps_tiles_gpu)

    
    # -------------------------
    # start going through tiles
    # -------------------------
   
    # initialize the image on the device
    im_gpu = drv.mem_alloc(image.astype(np.float32).nbytes)
    drv.memcpy_htod(im_gpu,image.astype(np.float32))
   

    # allocate key arrays -- these will be keys to sort particles into softening bins
    start_g.record()
    keys_gpu = drv.mem_alloc(int(4*hist.sum()))
    calculate_keys(ps_tiles_gpu, keys_gpu, np.int32(hist.sum()), np.float32(dx), 
                   block=(nthreads,1,1),grid=(32,1,1))
    end_g.record()
    end_g.synchronize()
    if timing: print '<<< Key generation took %f ms'%(start_g.time_till(end_g))

    keys = np.empty(hist.sum(), dtype=np.int32)


    # ----------------------------------------
    # sort particles by their softening length
    # ----------------------------------------
    for i in xrange(Ntiles) : 
        n_per_tile = tile_offsets[i+1] - tile_offsets[i]
        if n_per_tile > 0 : 
            radix_sort(int(keys_gpu), int(ps_tiles_gpu), tile_offsets[i], n_per_tile)

    drv.memcpy_dtoh(keys,keys_gpu)
    drv.memcpy_dtoh(ps_tiles,ps_tiles_gpu)
#    return keys,ps_tiles,tile_offsets,dx
        
    drv.Context.synchronize()

    tile_start = time.clock()
    for i in xrange(Ntiles) :
        n_per_tile = tile_offsets[i+1] - tile_offsets[i]
        if n_per_tile > 0 : 
            my_stream = streams[i%16]
            
            xmin_p, xmax_p, ymin_p, ymax_p  = tiles_physical[i]
            xmin_t, xmax_t, ymin_t, ymax_t  = tiles_pix[i]
            
            nx_tile = xmax_t-xmin_t+1
            ny_tile = ymax_t-ymin_t+1
                    
                
            # make everything the right size
            xmin_t,xmax_t,ymin_t,ymax_t = map(np.int32,[xmin_t,xmax_t,ymin_t,ymax_t])
            xmin_p,xmax_p,ymin_p,ymax_p = map(np.float32, [xmin_p,xmax_p,ymin_p,ymax_p])
            
            tile_render_kernel(ps_tiles_gpu,tile_offsets_gpu,np.int32(i),
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
    if timing: print '<<< %d tiles rendered in %f s'%(Ntiles,time.clock()-tile_start)

    drv.memcpy_dtoh(image,im_gpu)
    drv.stop_profiler()
    
    if timing: print '<<< Total render done in %f s\n'%(time.clock()-global_start)

    del(start_g)
    del(end_g)
    
    return image


def cu_template_render_image_single(s,nx,ny,xmin,xmax, qty='rho',timing = False, nthreads=128):
    """
    CPU part of the SPH render code that executes the rendering on the GPU
    
    does some basic particle set prunning and sets up the image
    tiles. It launches cuda kernels for rendering the individual sections of the image
    """
    import pycuda.driver as drv
    import pycuda.tools
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    from radix_sort import radix_sort

    global_start = time.clock()

    start = time.clock()
    # construct an array of particles
    Partstruct = [('x','f4'),('y','f4'),('qt','f4'),('h','f4')]
    ps = drv.pagelocked_empty(len(s),dtype=Partstruct)
    
    with s.immediate_mode : 
        ps['x'],ps['y'],ps['qt'],ps['h'] = [s[arr] for arr in ['x','y','mass','smooth']]

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

    start = time.clock()

    # ------------------
    # set up the kernels
    # ------------------
    code = file('/home/itp/roskar/homegrown/template_kernel.cu').read()
    mod = SourceModule(code)
    tile_histogram = mod.get_function("tile_histogram")
    distribute_particles = mod.get_function("distribute_particles")
    tile_render_kernel = mod.get_function("tile_render_kernel")
    calculate_keys = mod.get_function("calculate_keys")

    # allocate histogram array
    hist = np.zeros(Ntiles,dtype=np.int32)
    
    # transfer histogram array and particle data to GPU
    hist_gpu = drv.mem_alloc(hist.nbytes)
    drv.memcpy_htod(hist_gpu,hist)
    
    start_g = drv.Event()
    end_g = drv.Event()

    start_g.record()
    ps_on_gpu = drv.mem_alloc(ps_gpu.nbytes)
    drv.memcpy_htod(ps_on_gpu,ps_gpu)
    end_g.record()
    end_g.synchronize()

    if timing: print '<<< Particle copy onto GPU took %f ms'%(start_g.time_till(end_g))

    # make everything the right size
    xmin,xmax,ymin,ymax = map(np.float32, [xmin,xmax,ymin,ymax])
    nx,ny,Ntiles = map(np.int32, [nx,ny,Ntiles])

    start_g.record()
    tile_histogram(ps_on_gpu,hist_gpu,np.int32(len(ps_gpu)),xmin,xmax,ymin,ymax,nx,ny,Ntiles,
                   block=(nthreads,1,1),grid=(32,1,1))

    drv.Context.synchronize()
    drv.memcpy_dtoh(hist,hist_gpu)
    end_g.record()
    end_g.synchronize()
    if timing: print '<<< Tile histogram took %f ms'%(start_g.time_till(end_g))
    print "<<< Total particle array = %d"%(hist.sum())

    # ---------------------------------------------------------------------------------
    # figured out the numbers of particles per tile -- set up the tile particle buffers
    # ---------------------------------------------------------------------------------
    ps_tiles = np.empty(hist.sum(),dtype=Partstruct)
    ps_tiles_gpu = drv.mem_alloc(ps_tiles.nbytes)

    tile_offsets = np.array([0],dtype=np.int32)
    tile_offsets = np.append(tile_offsets, hist.cumsum().astype(np.int32))
    tile_offsets_gpu = drv.mem_alloc(tile_offsets.nbytes)
    drv.memcpy_htod(tile_offsets_gpu,tile_offsets)

    start_g.record()
    distribute_particles(ps_on_gpu, ps_tiles_gpu, tile_offsets_gpu, np.int32(len(ps_gpu)), 
                         xmin, xmax, ymin, ymax, nx, ny, Ntiles, 
                         block=(nthreads,1,1), grid=(np.int(Ntiles),1,1), shared=(nthreads*2+1)*4)
    end_g.record()
    end_g.synchronize()
    if timing: print '<<< Particle reshuffling took %f ms'%(start_g.time_till(end_g))
    drv.memcpy_dtoh(ps_tiles, ps_tiles_gpu)

    
    # -------------------------
    # start going through tiles
    # -------------------------
   
    # initialize the image on the device
    im_gpu = drv.mem_alloc(image.astype(np.float32).nbytes)
    drv.memcpy_htod(im_gpu,image.astype(np.float32))
   

    # allocate key arrays -- these will be keys to sort particles into softening bins
    start_g.record()
    keys_gpu = drv.mem_alloc(int(4*hist.sum()))
    calculate_keys(ps_tiles_gpu, keys_gpu, np.int32(hist.sum()), np.float32(dx), 
                   block=(nthreads,1,1),grid=(32,1,1))
    end_g.record()
    end_g.synchronize()
    if timing: print '<<< Key generation took %f ms'%(start_g.time_till(end_g))

    keys = np.empty(hist.sum(), dtype=np.int32)


    # ----------------------------------------
    # sort particles by their softening length
    # ----------------------------------------
    for i in xrange(Ntiles) : 
        n_per_tile = tile_offsets[i+1] - tile_offsets[i]
        if n_per_tile > 0 : 
            radix_sort(int(keys_gpu), int(ps_tiles_gpu), tile_offsets[i], n_per_tile)

    drv.memcpy_dtoh(keys,keys_gpu)
    drv.memcpy_dtoh(ps_tiles,ps_tiles_gpu)
#    return keys,ps_tiles,tile_offsets,dx
        
    drv.Context.synchronize()

    tile_start = time.clock()
    for i in xrange(Ntiles) :
        n_per_tile = tile_offsets[i+1] - tile_offsets[i]
        if n_per_tile > 0 : 
            my_stream = streams[i%16]
            
            xmin_p, xmax_p, ymin_p, ymax_p  = tiles_physical[i]
            xmin_t, xmax_t, ymin_t, ymax_t  = tiles_pix[i]
            
            nx_tile = xmax_t-xmin_t+1
            ny_tile = ymax_t-ymin_t+1
                    
                
            # make everything the right size
            xmin_t,xmax_t,ymin_t,ymax_t = map(np.int32,[xmin_t,xmax_t,ymin_t,ymax_t])
            xmin_p,xmax_p,ymin_p,ymax_p = map(np.float32, [xmin_p,xmax_p,ymin_p,ymax_p])
            
            tile_render_kernel(ps_tiles_gpu,tile_offsets_gpu,np.int32(i),
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
    if timing: print '<<< %d tiles rendered in %f s'%(Ntiles,time.clock()-tile_start)

    drv.memcpy_dtoh(image,im_gpu)
    drv.stop_profiler()
    
    if timing: print '<<< Total render done in %f s\n'%(time.clock()-global_start)

    del(start_g)
    del(end_g)
    
    return image


def test_template_render(s,nx,ny,xmin,xmax,qty='rho',sort_arr = None, timing = False) : 
    start = time.clock()
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]
    res = cu_template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax,sort_arr=sort_arr, timing=timing)
    print '<<< Done after %f seconds'%(time.clock()-start)
    return res 


