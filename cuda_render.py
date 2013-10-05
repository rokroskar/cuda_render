"""
Attempt to implement a CUDA-based SPH renderer
"""


from numbapro import vectorize
from numba import autojit, jit, double, int32
from numbapro import prange
import numpy as np
import pynbody
import scipy.integrate as integrate
#from numpy import int32  

#@jit('double(double,double)')
@vectorize([double(double,double)])
def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d<2 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
        
    return f/(np.pi*h**3)

#@autojit
#def _2D_kernel_func(d, h) : 
#    return 2*integrate.quad(lambda z : kernel_func(np.sqrt(z**2 + d**2),h),0,h)[0]

@jit('double(double,double,double)')
def distance(x,y,z) : 
    return np.sqrt(x**2+y**2+z**2)

@jit('int32(double,double,double)')
def physical_to_pixel(xpos,xmin,dx) : 
    return int32((xpos-xmin)/dx)

#@jit('double(double,double,double)')
def pixel_to_physical(xpix,x_start,dx) : 
    return dx*xpix+x_start

#@jit(double[:,:](double[:],double[:],double[:],double[:],double[:],double[:],double[:],int32,int32,double,double,double,double))
def render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax) : 
    MAX_D_OVER_H = 2.0

    image = np.zeros((nx,ny))

    Npart = len(xs)

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x_start = xmin+dx/2
    y_start = ymin+dy/2
    zplane = 0.0

    # set up the kernel values
    kernel_samples = np.arange(0,2.01,0.01)
    kernel_vals = kernel_func(kernel_samples,1.0)

    for i in xrange(Npart) : 

        x,y,z,h,qt = [double(xs[i]),double(ys[i]),double(zs[i]),double(hs[i]),double(qts[i]*mass[i]/rhos[i])]

        if h < dx*0.55 : h = dx*0.55

        # is the particle in the frame?
        if ((x > xmin-2*h) & (x < xmax+2*h) & 
            (y > ymin-2*h) & (y < ymax+2*h) & 
            (np.abs(z-zplane) < 2*h)) : 
        
            
            if (MAX_D_OVER_H*h/dx < 1 ) & (MAX_D_OVER_H*h/dy < 1) : 
                # pixel coordinates 
                xpos = physical_to_pixel(x,xmin,dx)
                ypos = physical_to_pixel(y,ymin,dy)
                # physical coordinates of pixel
                xpixel = pixel_to_physical(xpos,x_start,dx)
                ypixel = pixel_to_physical(ypos,y_start,dy)
                zpixel = zplane

                dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                d = distance(dxpix,dypix,dzpix)
                if (xpos > 0) & (xpos < nx) & (ypos > 0) & (ypos < ny) & (d/h < 2) : 
                    kernel_val = kernel_vals[int(d/(0.01*h))]/(h*h*h)
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
                                               
                        if (d/h < 2) : 
                            
                            kernel_val = kernel_vals[int32(d/(0.01*h))]/(h*h*h)
                            image[xpix,ypix] += qt*kernel_val

    return image

def start_image_render(s,nx,ny,xmin,xmax) : 
    # generate the rho and smooth arrays
    s['smooth']
    s['rho']

    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth','rho','mass','rho']]
    
    fast_render = jit('double[:,:](double[:],double[:],double[:],double[:],double[:],double[:],double[:],int32,int32,double,double,double,double)')(render_image)

    return fast_render(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax)
    

def try_image_render() : 
    x=y=z=hs=qts=mass=rhos= np.random.rand(100)
    nx=ny=10
    xmin=0
    xmax=1
    
    fast_render = jit('double[:,:](double[:],double[:],double[:],double[:],double[:],double[:],double[:],int32,int32,double,double,double,double)')(render_image)
    
    #render_image(x,y,z,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax)
    return fast_render(x,y,z,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax)
