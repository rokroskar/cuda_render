"""

A "template" sph renderer. Particles need to be sorted by smoothing
length when passed to the render routine. 

Some sample timings:

In [72]: %timeit numba_template_render.template_render_image(s.d,400,400,-.5,.5,-.5,.5,timing=True,two_d=0)
<<< Initial particle selection took 0.088440 s
<<< Bin sort done in 0.002324
<<< Rendering 21879 particles took 0.022327 s
<<< Total time: 0.114074 s
10 loops, best of 3: 118 ms per loop

In [73]: %timeit pynbody.sph.render_image(s.d,x1=-.5,x2=.5,nx=400,ny=400,threaded=False,approximate_fast=False,force_quiet=True,kernel=pynbody.sph.Kernel())
10 loops, best of 3: 87.5 ms per loop

Note that the actual render takes 22 ms and most of the time is spent doing the initial 
np.where and sort calls. 

Doing a line of sight integration through the whole volume:

In [74]: %timeit numba_template_render.template_render_image(s.d,400,400,-.5,.5,-.5,.5,timing=True,two_d=1)
<<< Initial particle selection took 0.130643 s
<<< Bin sort done in 0.139553
<<< Rendering 1293231 particles took 0.203697 s
<<< Total time: 0.485683 s
1 loops, best of 3: 499 ms per loop

In [75]: %timeit pynbody.sph.render_image(s.d,x1=-.5,x2=.5,nx=400,ny=400,threaded=False,approximate_fast=False,force_quiet=True,kernel=pynbody.sph.Kernel2D())
1 loops, best of 3: 862 ms per loop

and compared to the numba-fied simple renderer: 
In [77]: %timeit numba_sph_render.start_image_render(s.d,400,400,-.5,.5,two_d=1,timing=True)
1 loops, best of 3: 456 ms per loop


The differences become more apparent for larger numbers of pixels: 

In [78]: %timeit numba_template_render.template_render_image(s.d,1600,1600,-.5,.5,-.5,.5,timing=True,two_d=1)
<<< Initial particle selection took 0.133772 s
<<< Bin sort done in 0.199010
<<< Rendering 1293231 particles took 3.787264 s
<<< Total time: 4.133957 s
1 loops, best of 3: 4.15 s per loop

In [80]: %timeit numba_sph_render.start_image_render(s.d,1600,1600,-.5,.5,two_d=1)
	1 loops, best of 3: 6.05 s per loop

In [81]: %timeit pynbody.sph.render_image(s.d,x1=-.5,x2=.5,nx=1600,ny=1600,threaded=False,approximate_fast=False,force_quiet=True,kernel=pynbody.sph.Kernel2D())
Beginning SPH render at 0.10 s
Render done at 10.99 s
1 loops, best of 3: 10.7 s per loop

So both the numbified version of the normal pynbody sph renderer is quite a bit faster, 
but the template renderer seems to be more efficient still. On the 40M dark matter particles
of the Eris simulation this is pretty clear: 




"""

import numba
from numbapro import vectorize
from numba import autojit, jit, double, int32, void
import numpy as np
from numpy import mod,ceil,floor,sqrt
import time

@autojit(nopython=True)
def calculate_distance(template, dx, dy) : 
    side_length = template.shape[0]
    # where is the center position
    cen = side_length/2
    
    for i in range(side_length) : 
        for j in range(side_length) : 
            template[i,j] = sqrt(((i-cen)*dx)**2 + ((j-cen)*dy)**2)


@vectorize(['double(double,double)'])
def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d <= 2.0 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
   
    return f/(np.pi*h**3)

@jit('int32(double,double,double)',nopython=True)
def physical_to_pixel(xpos,xmin,dx) : 
    return int32((xpos-xmin)/dx)


def template_render_image(s,nx,ny,xmin,xmax,ymin,ymax,qty='rho',timing = False,two_d=0):
    """
    CPU part of the SPH render code
    
    does some basic particle set prunning and sets up the image
    tiles. It launches cuda kernels for rendering the individual sections of the image
    """
    
    time_init = time.clock()
    
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]

    # ----------------------
    # setup the global image
    # ----------------------
    image = np.zeros((nx,ny))
    
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    
    x_start = xmin+dx/2
    y_start = ymin+dy/2

    zplane = 0.0

    # ------------------------------------
    # trim particles based on image limits
    # ------------------------------------
    start = time.clock()
    ind = np.where((xs + 2*hs > xmin) & (xs - 2*hs < xmax) & 
                   (ys + 2*hs > ymin) & (ys - 2*hs < ymax) &
                   (np.abs(zs-zplane)*(1-two_d) < 2*hs))[0]

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

    # ---------------------
    # process the particles 
    # ---------------------
    start = time.clock()
    image = template_kernel_cpu(xs,ys,qts,hs,nx,ny,xmin,xmax,ymin,ymax,two_d)
    if timing: print '<<< Rendering %d particles took %f s'%(len(xs),
                                                             time.clock()-start)
    
    if timing: print '<<< Total time: %f s'%(time.clock()-time_init)

    return image

@jit('double[:,:](double[:],double[:],double[:],double[:],int32,int32,double,double,double,double, int32)')
def template_kernel_cpu(xs,ys,qts,hs,nx,ny,xmin,xmax,ymin,ymax,two_d) : 
    # ------------------
    # create local image 
    # ------------------
    image = np.zeros((nx,ny),dtype=np.float)

    Npart = len(hs) 
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

       
    # ------------------------------
    # start the loop through kernels
    # ------------------------------
    kmax = int(np.ceil(hs.max()*2.0/dx*2.0))
    kmin = int(np.floor(hs.min()*2.0/dx*2.0))
    # make sure kmin and kmax are odd
    if not mod(kmax,2) : kmax += 1
    if not mod(kmin,2) : kmin += 1
    kmin = max(1,kmin)
    kernel_base = np.ones((kmax,kmax))
    kernel = np.ones((kmax,kmax))
    calculate_distance(kernel_base,dx,dy)
    
    max_d_curr = 0.0
    start_ind = 0
    end_ind = 0
    for k in xrange(kmin,kmax+2,2) : 
        # ---------------------------------
        # the max. distance for this kernel
        # ---------------------------------
        max_d_curr = dx*np.floor(k/2.0)
        if max_d_curr < dx/2.0 : max_d_curr = dx/2.0

        i_max_d = double(1./max_d_curr)
        # -------------------------------------------------
        # find the chunk of particles that need this kernel
        # -------------------------------------------------
        for end_ind in xrange(start_ind,Npart) : 
            if 2*hs[end_ind] < max_d_curr : pass
            else: break
        
        Nper_kernel = end_ind-start_ind
        
        # -------------------------------------------------------------------------
        # only continue with kernel generation if there are particles that need it!
        # -------------------------------------------------------------------------
        if Nper_kernel > 0 : 
            kernel = kernel_base[kmax/2-k/2:kmax/2+k/2+1,
                                 kmax/2-k/2:kmax/2+k/2+1]
            kernel = kernel_func(kernel*i_max_d*2.0,1.0)
            kernel *= 8*i_max_d*i_max_d*i_max_d # kernel / h**3
        
            # --------------------------------
            # paint each particle on the image
            # --------------------------------
            for pind in xrange(start_ind,end_ind) : 
                x,y,h,qt = [xs[pind],ys[pind],hs[pind],qts[pind]]
                
                # set the minimum h to be equal to half pixel width
                #                h = max_d_curr*.5
                #h = max(h,0.55*dx)
                
                # particle pixel center
                xpos = physical_to_pixel(x,xmin,dx)
                ypos = physical_to_pixel(y,ymin,dy)
    
                left  = xpos-k/2
                right = xpos+k/2+1
                upper = ypos-k/2
                lower = ypos+k/2+1

                ker_left = abs(min(left,0))
                ker_right = k + min(nx-right,0)
                ker_upper = abs(min(upper,0))
                ker_lower = k + min(ny-lower,0)
                
                for i in xrange(0,k) : 
                    for j in xrange(0,k): 
                        if ((i+left>=0) and (i+left < nx) and (j+upper >=0) and (j+upper<ny)) : 
                            image[(i+left),(j+upper)] += kernel[i,j]*qt


            start_ind = end_ind

    return image
