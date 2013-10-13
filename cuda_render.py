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

@vectorize(['double(double,double)'])
def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d <= 2.0 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
   
    return f/(np.pi*h**3)

@vectorize([double(double,double)])
def kernel_func_norm(d, h) : 
    dnorm = d/h
    if dnorm < 1 : 
        f = 1.-(3./2)*dnorm**2 + (3./4.)*dnorm**3
    elif dnorm <= 2.0 :
        f = 0.25*(2.-dnorm)**3
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

@jit('double(int32,double,double)')
def pixel_to_physical(xpix,x_start,dx) : 
    return dx*xpix+x_start

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


def start_image_render(s,nx,ny,xmin,xmax) : 
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth','rho','mass','rho']]
    return render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax)



############################
#                          #
#  TEMPLATE RENDERING CODE #
#                          #
############################

@cuda.jit('int32(double,double,double)',nopython=True,device=True,inline=True)
def cu_physical_to_pixel(xpos,xmin,dx) : 
    return int32((xpos-xmin)/dx)

@cuda.jit('double(int32,double,double)',nopython=True, device=True, inline=True)
def cu_pixel_to_physical(xpix,x_start,dx) : 
    return dx*xpix+x_start


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
        return np.ones((sqrt(Ntotal),sqrt(Ntotal)),dtype=np.float)
    
    else : 
        template = np.zeros((Ntemplate,Ntemplate),dtype=np.float)-1
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

@autojit
def test(N):
    return mod(Ntemplate,2)

#@jit(float32[:,:](int32))
#s1rt(0@autojit
def cu_make_template(k) : 
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
        template = np.ones((sqrt(Ntotal),sqrt(Ntotal)))
    
    else : 
        template = np.zeros((Ntemplate,Ntemplate))
        template -= 1
        Nbase = Ntemplate-2
        Nleft = Ntotal - Nbase**2

        # left-overs must be divisible by 4 and odd
        if (mod(Nleft,4) != 0) or (mod(Nleft/4,2) != 1) :
            return template
        
        else :
            # set the base to 1
            template[1:-1,1:-1] = 1
            
            # set up the outer pixels
            template[(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2,0] = 1
            template[(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2,-1] = 1
            template[0,(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2] = 1
            template[-1,(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2] = 1
        
    return template

#
@autojit(nopython=True)
def cu_calculate_distance(template, dx, dy) : 
    side_length = template.shape[0]
    # where is the center position
    cen = side_length/2
    
    for i in xrange(side_length) : 
        for j in xrange(side_length) : 
            template[i,j] = sqrt(((i-cen)*dx)**2 + ((j-cen)*dy)**2)
    

@autojit
def get_tile_ids(tiles_physical,xmin,ymin,xs,ys,tileids) : 
    t_dx = tiles_physical[0,1]-tiles_physical[0,0]
    t_dy = tiles_physical[0,3]-tiles_physical[0,2]
    
    x_ind = np.floor((xs-xmin)/t_dx)
    y_ind = np.floor((ys-ymin)/t_dy)
    tileids[:] = x_ind*(np.sqrt(tiles_physical.shape[0]))+y_ind

@autojit(nopython=True)
def calculate_distance(template, dx, dy) : 
    side_length = template.shape[0]
    # where is the center position
    cen = side_length/2
    
    for i in range(side_length) : 
        for j in range(side_length) : 
            template[i,j] = sqrt(((i-cen)*dx)**2 + ((j-cen)*dy)**2)



def generate_template_set(kmax) : 
    ts = []
    ds = []
    ks = np.arange(kmax,dtype=np.float32)

    for k in ks :
        try: 
            newt = make_template(k)
            # append the new template
            ts.append(newt)
            ds.append(calculate_distance(newt.copy(),1.0,1.0).max())
        except RuntimeError: 
            pass
    
    ds = np.array(ds)
    ts = np.array(ts)

    # how many unique template/distance pairs
    ds, ind = np.unique(ds,return_index=True)

    ts = ts[ind]
    ks = ks[ind]
    np.savez('image_templates',ts=ts,ds=ds,ks=ks)
    print 'found %d unique templates'%len(ds)
    return ts, ds, ks

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

@autojit
def template_kernel_cpu(xs,ys,qts,hs,nx,ny,xmin,xmax,ymin,ymax) : 
    # ****************************************************
    # copy kernels to shared memory for faster access here
    # ****************************************************

    # create local image 
    # ** this will be done in shared memory **
    image = np.zeros((nx,ny),dtype=np.float)

    Npart = len(hs) 
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

       
    # determine which particles this thread should process
    Nthreads = 1#cuda.gridDim.x * cuda.blockDim.x
    my_thread_id = 0#cuda.grid(1)
    
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
    calculate_distance(kernel_base,dx,dy)
#    print 'KMAX, KMIN', kmax, kmin
    
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
        for j in xrange(start_ind,Npart) : 
            if 2*hs[j] < max_d_curr : pass
            else: break

        end_ind = j
        
        Nper_kernel = end_ind-start_ind
        
        # -------------------------------------------------------------------------
        # only continue with kernel generation if there are particles that need it!
        # -------------------------------------------------------------------------
        if Nper_kernel > 0 : 
            kernel = kernel_base[kmax/2-k/2:kmax/2+k/2+1,
                                 kmax/2-k/2:kmax/2+k/2+1]
            kernel = kernel_func(kernel*i_max_d*2.0,1.0)
            kernel *= i_max_d*i_max_d*i_max_d
#            print 'Processing %d particles for k = %d'%(end_ind-start_ind, k)
        
            # --------------------------------------
            # determine thread particle distribution
            # --------------------------------------
            Nper_thread = Nper_kernel/Nthreads
            n_start = Nper_thread*my_thread_id+start_ind

            # if this is the last thread, make it pick up the slack
            if my_thread_id == Nthreads-1 : 
                n_end = end_ind
            else : 
                n_end = Nper_thread*(my_thread_id+1)+n_start
                    
            # all threads have their particle indices figured out, increment for next iteration
            start_ind = end_ind

            # print 'nperthread = ', Nper_thread, 'n_start = ', n_start, 'n_end = ', n_end
                #print 'Thread %d processing %d particles for k = %d'%(my_thread_id,n_end-n_start,k)

            # ------------------------
            # synchronize threads here
            # ------------------------
        
            # cuda.syncthreads()
        
            # --------------------------------
            # paint each particle on the image
            # --------------------------------
            for pind in xrange(n_start,n_end) : 
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
                
                ker_val = kernel[ker_left:ker_right,ker_upper:ker_lower]
                ker_val *= qt

                image[max(left,0):min(right,nx),max(upper,0):min(lower,nx)] += ker_val 
            
            # --------------------------------
            # check if we have reached the end
            # --------------------------------
            if end_ind == Npart-1 : 
                break

    return image

def cu_template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax, sort_arr = None, timing = False):
    """
    CPU part of the SPH render code
    
    does some basic particle set prunning and sets up the image
    tiles. It launches cuda kernels for rendering the individual sections of the image
    """
    
    from bisect import bisect

    # ----------------------
    # setup the global image
    # ----------------------
    #image,dx,dy,x_start,y_start,pix_kernels,ds,ks=setup_template_image(xs,ys,zs,hs,qts,mass,rhos,
     #nx,ny,xmin,xmax,ymin,ymax,debug)
    
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
    ind = np.where((np.abs(zs-zplane) < 2*hs) & 
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
    # in this step only process particles that need kernels < 50 pixels
    # -----------------------------------------------------------------
    
    start = time.clock()
    tiles_pix, tiles_physical = make_tiles(nx,ny,xmin,xmax,ymin,ymax,100)
    if timing: print '<<< Tiles made in %f s'%(time.clock()-start)

    start = time.clock()
    ind = np.searchsorted(hs,[0.0,25.*dx/2.0]) # indices of particles with 2h/dx < 50 pixels
    if timing: print '<<< Sorted search took %f s'%(time.clock()-start)
    
    start = time.clock()
    process_tiles(xs[:ind[1]],ys[:ind[1]],
                  qts[:ind[1]],hs[:ind[1]],tiles_pix,tiles_physical,image)    
    if timing: print '<<< Processing %d tiles with %d particles took %f s'%(len(tiles_pix),
                                                                            ind[1],time.clock()-start)
    # --------------------------------------------------
    # process the particles with large smoothing lengths
    # --------------------------------------------------
    if ind[1] != len(xs) : 
        start = time.clock()
        image += template_kernel_cpu(xs[ind[1]:],ys[ind[1]:],qts[ind[1]:],hs[ind[1]:],nx,ny,xmin,xmax,ymin,ymax)
        if timing: print '<<< Processing %d particles with large smoothing lengths took %f s'%(len(xs)-ind[1],
                                                                                               time.clock()-start)

    return image, xs,ys,qts,hs

#@jit(void(double[:],double[:],double[:],double[:],int32[:,:],double[:,:],double[:,:]))
@autojit
def process_tiles(xs,ys,qts,hs,tiles_pix,tiles_physical,image):

    Ntiles = tiles_pix.shape[0]

    for i in xrange(Ntiles) :
        
        tile   = tiles_pix[i]
        tile_p = tiles_physical[i]

        tile_xmin, tile_xmax, tile_ymin, tile_ymax  = tile_p[0],tile_p[1],tile_p[2],tile_p[3]
        
        nx_tile = tile[1]-tile[0]+1
        ny_tile = tile[3]-tile[2]+1
         
#        inds = np.arange(xs.shape[0])
        inds = np.where((xs + 2*hs > tile_xmin) & (xs - 2*hs < tile_xmax) & 
                        (ys + 2*hs > tile_ymin) & (ys - 2*hs < tile_ymax))[0]                      

     #   print '<<< Tile %d where took %f s'%(i,time.clock()-start)
 #       print 'Starting tile %d with %d particles'%(i,imax-imin)
        #print 'Tile limits: ',tile_p

        if inds.shape[0] > 0 : 
#            start = time.clock()
            im_tile = template_kernel_cpu(xs[inds],ys[inds],qts[inds],hs[inds],
                                          nx_tile,ny_tile,tile_xmin,tile_xmax,tile_ymin,tile_ymax)
            image[tile[0]:tile[1]+1,tile[2]:tile[3]+1] += im_tile
 #           if timing: print '<<< Tile %d render took %f s'%(i,time.clock()-start)


@autojit
def where_numba(x) : 
    return np.where(x>0)
    
def where_nonumba(x):
    return np.where(x>0)

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

def setup_template_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax, debug = False):
    from bisect import bisect
        
    # ----------------
    # image parameters
    # ----------------

    MAX_D_OVER_H = 2.0

    image = np.zeros((nx,ny))

    Npart = len(xs)

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    
    x_start = xmin+dx/2
    y_start = ymin+dy/2

    zplane = 0.0

    # -----------------------------------------------------
    # load or generate a template library and an array of
    # max. physical distance corresponding to each template
    # -----------------------------------------------------

    try : 
        dat = np.load('image_templates.npz')
        ts = dat['ts']
        ds = dat['ds']
        ks = dat['ks']

    except IOError : 
        'Templates not found -- generating a new set'
        ts, ds, ks = generate_template_set(500)
    
    del(ds)
    ds = []
    
    for i,t in enumerate(ts) : 
        calculate_distance(t,dx,dy)
        # store the max distance
        ds.append(t.max())
        # normalize and apply kernel function
        ts[i] = kernel_func(t/t.max()*2.0,1.0)

    ds = np.array(ds)
    
    
    return image, dx, dy, x_start, y_start, ts, ds


def template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax, debug = False):
    from bisect import bisect
    zplane = 0.0

    image,dx,dy,x_start,y_start,ts,ds=setup_template_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax, debug)
    
    # trim particles based on image limits
    ind = np.where((xs > xmin-2*hs) & (xs < xmax+2*hs) & 
                   (ys > ymin-2*hs) & (ys < ymax+2*hs) & 
                   (np.abs(zs-zplane) < 2*hs))[0]
    xs,ys,zs,hs,qts,mass,rhos = (xs[ind],ys[ind],zs[ind],hs[ind],qts[ind],mass[ind],rhos[ind])

    # ---------------------------------------------------------------
    # calculate which template ('k') should be used for each particle
    # and sort particles by k
    # ---------------------------------------------------------------

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
            if debug : 
                print 'rendering particles in dbin = ', dind
                print 'd = ', ds[dind], 'min(h) = %f max(h) = %f'%(min(hs[sl]),max(hs[sl]))
                
            template_kernel(xs[sl],
                            ys[sl],
                            zs[sl],
                            hs[sl],
                            qts[sl],
                            nx,ny,xmin,xmax,xmin,xmax,image,ts[dind])
        
    return image, ts
    
    
def test_template_render(s,nx,ny,xmin,xmax,qty='rho',sort_arr = None, timing = False) : 
    start = time.clock()
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]
    res = cu_template_render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax,sort_arr=sort_arr, timing=timing)
    print '<<< Done after %f seconds'%(time.clock()-start)
    return res 


