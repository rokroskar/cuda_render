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

@vectorize([double(double,double)])
def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d <= 2 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
    if d < 0 : 
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

#@cuda.jit(void(double[:,:,:],                           # positions
#               double[:],double[:],double[:],double[:], # hs, qts, mass, rhos
#               int32,int32,double,double,double,double, # nx, ny, xmin, xmax, ymin, ymax
#               double[:,:],                             # image
#               double[:]))                              # kernel_vals
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


#@cuda.jit(void(numba.int32[:]))
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

@autojit
def template_kernel(xs,ys,zs,hs,qts,nx,ny,xmin,xmax,ymin,ymax,image,kernel) : 
    Npart = len(hs) 
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    kernel_dim = kernel.shape[0]

    # paint each particle on the image
    for i in xrange(Npart) : 
        x,y,z,h,qt = [xs[i],ys[i],zs[i],hs[i],qts[i]]

        # particle pixel center
        xpos = physical_to_pixel(x,xmin,dx)
        ypos = physical_to_pixel(y,ymin,dy)
    
        left  = xpos-kernel_dim/2
        right = xpos+kernel_dim/2+1
        upper = ypos-kernel_dim/2
        lower = ypos+kernel_dim/2+1

        ker_left = abs(min(left,0))
        ker_right = kernel_dim + min(nx-right,0)
        ker_upper = abs(min(upper,0))
        ker_lower = kernel_dim + min(ny-lower,0)
        image[max(left,0):min(right,nx),
              max(upper,0):min(lower,nx)] += kernel[ker_left:ker_right,
                                                        ker_upper:ker_lower]*qt/(h*h*h)

def template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax):
    from bisect import bisect
    # image parameters
    MAX_D_OVER_H = 2.0

    image = np.zeros((nx,ny))

    Npart = len(xs)

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x_start = xmin+dx/2
    y_start = ymin+dy/2
    
    zplane = 0.0
    zpixel = zplane


    # generate a template library and an array of max. physical
    # distance corresponding to each template

    ts = []
    ds = []

    for k in range(5000) :
        try: 
            newt = calculate_distance(make_template(k), dx = dx, dy = dy)
            # store the max distance
            ds.append(newt.max())
            # append the new template, normalized and run through the kernel function
            newt = newt/newt.max()*2.0
            ts.append(kernel_func(newt,1.0))

        except RuntimeError: 
            pass

    ds = np.array(ds)
    ts = np.array(ts)
    
    # how many unique templates do we have? 
    ds, ind = np.unique(ds,return_index=True)
    ts = ts[ind]

    # trim particles based on image limits
    ind = np.where((xs > xmin-2*hs) & (xs < xmax+2*hs) & 
                   (ys > ymin-2*hs) & (ys < ymax+2*hs) & 
                   (np.abs(zs-zplane) < 2*hs))[0]
    xs,ys,zs,hs,qts,mass,rhos = (xs[ind],ys[ind],zs[ind],hs[ind],qts[ind],mass[ind],rhos[ind])
    print len(ind)
    print xs.max(), xs.min(), ys.max(), ys.min(), zs.max(), zs.min(), hs.max(), hs.min()
    # calculate which template ('k') should be used for each particle
    # and sort particles by k
    max_d = 2.0*hs
    dbin = np.digitize(max_d,ds)-1
    dbin_sortind = dbin.argsort()
    dbin_sorted = dbin[dbin_sortind]
    
    
    # set the render quantity 
    qts = qts*mass/rhos

    # call the render kernel for each template bin
    prev_index = min(dbin)
    xs,ys,zs,hs,qts = (xs[dbin_sortind],ys[dbin_sortind],zs[dbin_sortind],hs[dbin_sortind],qts[dbin_sortind])
    for dind in xrange(min(dbin),max(dbin)) : 
        
        next_index = bisect(dbin_sorted,dind)
        if next_index != prev_index : 
            sl = slice(prev_index,next_index)
            prev_index = next_index
            print 'rendering particles in dbin = ', dind
            print 'd = ', ds[dind], 'min(h) = %f max(h) = %f'%(min(hs[sl]),max(hs[sl]))
            template_kernel(xs[sl],
                            ys[sl],
                            zs[sl],
                            hs[sl],
                            qts[sl],
                            nx,ny,xmin,xmax,xmin,xmax,image,ts[dind])
        
    return image, ds, ts, dbin, dbin_sortind, dbin_sorted
    
    
def test_template_render(s,nx,ny,xmin,xmax,qty='rho') : 
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]
    return template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax)


#############################
# render template functions #
#############################

from numpy import ceil, floor, sqrt, mod

def make_template(k) : 
    # total number of cells we need
    Ntotal = 1 + 4*k

    # total number of cells required for the template
    Ntemplate = ceil(sqrt(Ntotal))

    # need an odd number of cells
    if mod(Ntemplate,2) == 0 : 
        Ntemplate = Ntemplate + 1

    # number of cells in the base template -- if the number of total
    # cells equals the number of template cells, we're done

    if sqrt(Ntotal) == Ntemplate : 
        return np.ones((sqrt(Ntotal),sqrt(Ntotal)),dtype=np.float32)
    
    else : 
        template = np.zeros((Ntemplate,Ntemplate),dtype=np.float32)-1
        Nbase = Ntemplate-2
    
        # set the base to 1
        template[1:-1,1:-1] = 1
    
        Nleft = Ntotal - Nbase**2
        
        # left-overs must be divisible by 4 and odd
        if (mod(Nleft,4) != 0) or (mod(Nleft/4,2) != 1) :
            raise(RuntimeError)

        template[(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2,0] = 1
        template[(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2,-1] = 1
        template[0,(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2] = 1
        template[-1,(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2] = 1
        
        return template


def calculate_distance(template, dx = 1.0, dy = 1.0, normalize = None) : 
    side_length = template.shape[0]
    # where is the center position
    cen = floor(side_length/2)
    
    for i in range(side_length) : 
        for j in range(side_length) : 
            template[i,j] *= sqrt(((i-cen)*dx)**2 + ((j-cen)*dy)**2)

    if normalize is not None : 
        template = template/template.max()*normalize
        return template
    else : 
        return template


