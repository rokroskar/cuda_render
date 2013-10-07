"""
Attempt to implement a CUDA-based SPH renderer
"""

import numba
from numbapro import vectorize, cuda
from numba import autojit, jit, double, int32, void
from numbapro import prange
import numpy as np
import pynbody
import scipy.integrate as integrate
import math

@vectorize([double(double,double)])
def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d<2 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
        
    return f/(np.pi*h**3)

@autojit
def _2D_kernel_func(d, h) : 
    return 2*integrate.quad(lambda z : kernel_func(np.sqrt(z**2 + d**2),h),0,h)[0]

@autojit
def distance(x,y,z) :
    return np.sqrt(x*x+y*y+z*z)

@jit('int32(double,double,double)',nopython=True)
def physical_to_pixel(xpos,xmin,dx) : 
    return int32((xpos-xmin)/dx)

@jit('double(int32,double,double)',nopython=True)
def pixel_to_physical(xpix,x_start,dx) : 
    return dx*xpix+x_start

#@jit(double[:,:](double[:],double[:],double[:],double[:],double[:],double[:],double[:],int32,int32,double,double,double,double))
def render_using_single_particle(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax) : 
    Npart = len(xs)

    image = np.zeros((nx,ny),dtype=np.double)

    # set up the kernel values
    kernel_samples = np.arange(0,2.01,0.01,dtype=np.float)
    kernel_vals = kernel_func(kernel_samples,1.0)

 #   for i in xrange(Npart) : 
  #      x,y,z,h,qt = [double(xs[i]),double(ys[i]),double(zs[i]),double(hs[i]),double(qts[i]*mass[i]/rhos[i])]
#        render_single_particle(x,y,z,h,qt,nx,ny,xmin,xmax,ymin,ymax,image,kernel_vals)
    
    return image

#@jit(void(double,double,double,double,double,int32,int32,double,double,double,double,double[:,:],double[:]))
def render_single_particle(x,y,z,h,qt,nx,ny,xmin,xmax,ymin,ymax,image,kernel_vals) :
    MAX_D_OVER_H = 2.0

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x_start = xmin+dx/2
    y_start = ymin+dy/2
    zplane = 0.0

    
    # is the particle in the frame?
    if ((x > xmin-2*h) & (x < xmax+2*h) & 
        (y > ymin-2*h) & (y < ymax+2*h) & 
        (np.abs(z-zplane) < 2*h)) : 

        if h < dx*0.55 : h = dx*0.55        
            
        if (MAX_D_OVER_H*h/dx < 1 ) & (MAX_D_OVER_H*h/dy < 1) : 
            # pixel coordinates 
            xpos = int32(physical_to_pixel(x,xmin,dx))
            ypos = int32(physical_to_pixel(y,ymin,dy))
            # physical coordinates of pixel
            xpixel = pixel_to_physical(xpos,x_start,dx)
            ypixel = pixel_to_physical(ypos,y_start,dy)
            zpixel = zplane

            dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
            d = distance(dxpix,dypix,dzpix)
            if (xpos > 0) & (xpos < nx) & (ypos > 0) & (ypos < ny) & (d/h < 2) : 
                kernel_val = kernel_vals[int32(d/(0.01*h))]/(h*h*h)
                image[xpos,ypos] += qt*kernel_val

        else :
            # bottom left of pixels the particle will contribute to
            x_pix_start = int32(physical_to_pixel(x-MAX_D_OVER_H*h,xmin,dx))
            y_pix_start = int32(physical_to_pixel(y-MAX_D_OVER_H*h,ymin,dy))
            # upper right of pixels the particle will contribute to
            x_pix_stop  = int32(physical_to_pixel(x+MAX_D_OVER_H*h,xmin,dx))
            y_pix_stop  = int32(physical_to_pixel(y+MAX_D_OVER_H*h,ymin,dy))
            
            if(x_pix_start < 0):  x_pix_start = 0
            if(x_pix_stop  > nx): x_pix_stop  = int32(nx-1)
            if(y_pix_start < 0):  y_pix_start = 0
            if(y_pix_stop  > ny): y_pix_stop  = int32(ny-1)
    


            for xpix in range(x_pix_start, x_pix_stop) : 
                for ypix in range(y_pix_start, y_pix_stop) : 
                    # physical coordinates of pixel
                    xpixel = pixel_to_physical(xpix,x_start,dx)
                    ypixel = pixel_to_physical(ypix,y_start,dy)
                    zpixel = zplane

                    dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                    d = distance(dxpix,dypix,dzpix)
                                               
                    if d/h < 2 : 
                        kernel_val = kernel_vals[int32(d/(0.01*h))]/(h*h*h)
                        image[xpix,ypix] += qt*kernel_val

@jit(double[:,:](double[:],double[:],double[:],double[:],double[:],double[:],double[:],int32,int32,double,double,double,double))
def render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax) : 
    MAX_D_OVER_H = 2.0

    image = np.zeros((nx,ny))

    Npart = len(xs)

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x_start = xmin+dx/2
    y_start = ymin+dy/2
    zplane = 0.0
    zpixel = zplane

    # set up the kernel values
    kernel_samples = np.arange(0,2.01,0.01,dtype=np.float)
    kernel_vals = kernel_func(kernel_samples,1.0)

    for i in xrange(Npart) : 
        x,y,z,h,qt = [double(xs[i]),double(ys[i]),double(zs[i]),double(hs[i]),double(qts[i]*mass[i]/rhos[i])]

        # is the particle in the frame?
        if ((x > xmin-2*h) and (x < xmax+2*h) and 
            (y > ymin-2*h) and (y < ymax+2*h) and 
            (np.abs(z-zplane) < 2*h)) : 

            if h < dx*0.55 : h = dx*0.55        
            
            h3 = 1./(h*h*h)
            hsq = h*h
            h1 = 1./(.01*h)

            if (MAX_D_OVER_H*h/dx < 1 ) and (MAX_D_OVER_H*h/dy < 1) : 
                # pixel coordinates 
                xpos = int32(physical_to_pixel(x,xmin,dx))
                ypos = int32(physical_to_pixel(y,ymin,dy))
                # physical coordinates of pixel
                xpixel = pixel_to_physical(xpos,x_start,dx)
                ypixel = pixel_to_physical(ypos,y_start,dy)

                dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                dsq = dxpix*dxpix + dypix*dypix + dzpix*dzpix 

                if (xpos > 0) and (xpos < nx) and (ypos > 0) and (ypos < ny) and (dsq/hsq < 4) : 
                    image[ypos,xpos] += qt*kernel_vals[int32(np.sqrt(dsq)*h1)]*h3

            else :
                # bottom left of pixels the particle will contribute to
                x_pix_start = physical_to_pixel(x-MAX_D_OVER_H*h,xmin,dx)
                y_pix_start = physical_to_pixel(y-MAX_D_OVER_H*h,ymin,dy)
                # upper right of pixels the particle will contribute to
                x_pix_stop  = physical_to_pixel(x+MAX_D_OVER_H*h,xmin,dx)
                y_pix_stop  = physical_to_pixel(y+MAX_D_OVER_H*h,ymin,dy)
            
                if(x_pix_start < 0):  x_pix_start = 0
                if(x_pix_stop  > nx): x_pix_stop  = int32(nx-1)
                if(y_pix_start < 0):  y_pix_start = 0
                if(y_pix_stop  > ny): y_pix_stop  = int32(ny-1)
                
                for xpos in xrange(x_pix_start, x_pix_stop) :
                    # physical coordinates of pixel
                    xpixel = pixel_to_physical(xpos,x_start,dx)
                    
                    for ypos in xrange(y_pix_start, y_pix_stop) : 
                        # physical coordinates of pixel
                        ypixel = pixel_to_physical(ypos,y_start,dy)
                        dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                        dsq = dxpix*dxpix+dypix*dypix+dzpix*dzpix
                        
                        if dsq/hsq < 4 : 
                            image[ypos,xpos] +=qt*kernel_vals[int32(np.sqrt(dsq)*h1)]*h3
                                                              
    return image

@cuda.jit(void(double[:,:,:],                           # positions
               double[:],double[:],double[:],double[:], # hs, qts, mass, rhos
               int32,int32,double,double,double,double, # nx, ny, xmin, xmax, ymin, ymax
               double[:,:],                             # image
               double[:]))                              # kernel_vals
def render_image_cuda(pos,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax,image,kernel_vals) : 
    MAX_D_OVER_H = 2.0

    Npart = pos.shape[0]

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x_start = xmin+dx/2.
    y_start = ymin+dy/2.
    zplane = 0.0
    zpixel = zplane
        
    # thread ID
    i = cuda.grid(1)
    
    # which set of particles should this thread work on?
    part_per_thread = int32(Npart/cuda.blockDim.x)
    my_parts = part_per_thread

    # if this is the last thread, pick up the slack
    if i == cuda.blockDim.x : my_parts = (Npart-part_per_thread*cuda.blockDim.x) + part_per_thread
    shared = cuda.shared.array(shape=(10),dtype=double)

    cuda.syncthreads()

    shared[0] += 1.0

    cuda.syncthreads()

    image[0,0] = shared[0]


@cuda.jit(void(numba.int32[:]))
def sum_cu(x) :
    shm = cuda.shared.array(shape=(1),dtype=numba.int32)
    cuda.atomic.add(shm,0,1)
    cuda.syncthreads()
#    if cuda.threadIdx.x == 0 : x[cuda.blockIdx.x] = shm[0]
    cuda.atomic.add(x,0,1)
    
def sum_test() :
    Nblocks  = 32
    Nthreads = 1024
    
    x = np.zeros((Nblocks),dtype=np.int)
    d_x = cuda.to_device(x)
    sum_cu[Nblocks,Nthreads](d_x)
    d_x.to_host()
    return x

def start_cuda_image_render(s,nx,ny,xmin,xmax,qty='rho') : 
    pos,hs,qts,mass,rhos = [s[arr] for arr in ['pos','smooth',qty,'mass','rho']]
    
    stream = cuda.stream()

    # send everything to device
    d_pos = cuda.to_device(pos,stream=stream)
    d_hs = cuda.to_device(hs,stream=stream)
    d_qts = cuda.to_device(qts,stream=stream)
    d_mass = cuda.to_device(mass,stream=stream)
    d_rhos = cuda.to_device(rhos,stream=stream)

    # set up the image array and send it to device
    image = np.zeros((nx,ny),dtype=float)
    d_image = cuda.to_device(image,stream=stream)
    
    # set up the kernel values and send them to device
    kernel_samples = np.arange(0,2.01,0.01,dtype=np.float)
    kernel_vals = kernel_func(kernel_samples,1.0)
    d_kernel_vals = cuda.to_device(kernel_vals,stream=stream)
    
    griddim = 1
    blockdim = 32
    render_image_cuda[griddim,blockdim](d_pos,d_hs,d_qts,d_mass,d_rhos,nx,ny,xmin,xmax,xmin,xmax,d_image,d_kernel_vals)
    
    d_image.to_host()


    return image

def start_image_render(s,nx,ny,xmin,xmax) : 
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth','rho','mass','rho']]
    return render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax)

def try_image_render() : 
    x=y=z=hs=qts=mass=rhos= np.random.rand(100)
    nx=ny=10
    xmin=0
    xmax=1
    
    render_image_serial(x,y,z,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax)

import numbapro

@autojit
def privatization_rules():
    reduction = 1.0
    private = 2.0
    shared = 3.0
    for i in numbapro.prange(10):
        reduction += i      # The inplace operator specifies a sum reduction
        reduction -= 1
        reduction *= 4      # ERROR: inconsistent reduction operator!
                            # '*' is a product reduction, not a sum reduction
        print 'reduction', reduction, i 

        print private       # ERROR: private is not yet initialized!
        private = i * 4.0   # This assignment makes it private
        print private       # Private is available now, this is fine

        print shared        # This variable is only ever read, so it's shared

    print 'reduction = ', reduction         # prints the sum-reduced value
    print 'private = ', private           # prints the last value, i.e. 99 * 4.0

