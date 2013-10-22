#include <stdio.h>
#include <math.h>

#define PI 3.141592654f
#define PI_I 1.0/PI

void kernel_func(double *, double, double, int);
void kernel_distance(double *, double, double, int);
void tile_render_kernel(double *, double *, double *, double *, int,
			int, int, int, int, int, int, double *, int, int);

void kernel_func(double *kernel, double h, double max_d, int ksize)
{
  /* 
     calculate the kernel value given the distance from center 
     normalized to 2*smoothing_length
  */
  double d;  

  for(int i=0;i<ksize*ksize;i++) 
    {
      d = kernel[i]/(max_d/2.0);
      kernel[i] = 0.0;
      if(d < 1) kernel[i] = (1.-(3./2)*(d*d) + (3./4.)*(d*d*d))*PI_I/(h*h*h);
      else if (d <= 2.0) kernel[i] = 0.25*powf((2.-d),3)*PI_I/(h*h*h); 
    }
}

void kernel_distance(double *kernel,double dx,double dy,int ksize) 
{
  /*
    calculate distance of pixel from center in physical units
  */

  int cen;
  int x,y;
  double dxpix,dypix;
  cen = ksize/2;
  
  for(int i=0;i<ksize*ksize;i++) 
    {
      x = i/ksize;
      y = i - x*ksize;
      
      dxpix = (double)(x-cen)*dx;
      dypix = (double)(y-cen)*dy;
      kernel[y*ksize+x] = sqrtf(dxpix*dxpix + dypix*dypix);
    }
}

/*    not needed in single cpu code, but might be useful when parallelized? 

void update_image(double *global, double *local, int x_offset, int y_offset, int nx_glob, int nx_loc, int ny_loc)
{
  int idx = threadIdx.x;
  int pix_per_thread = nx_loc*ny_loc/(blockDim.x);

  int my_start, my_end, loc_x, loc_y;
  
  my_start = idx*pix_per_thread;
  my_end = my_start + pix_per_thread;
  // if this is the last thread, make it take the rest of the pixels
  if (idx == blockDim.x*blockIdx.x-1) my_end = nx_loc*ny_loc;
  
  for(int p = my_start; p < my_end; p++) 
    {
      loc_y = p/nx_loc;
      loc_x = p - loc_y*nx_loc;
      global[(loc_x+x_offset) + (loc_y+y_offset)*nx_glob] += local[p];
    }
}
*/
void tile_render_kernel(double *xs, double *ys, double *qts, double *hs, int Npart,
			int kmin, int kmax, int xmin, int xmax, int ymin, int ymax, 
			double *image, int nx, int ny)
{    
  
  double kernel[kmax*kmax];

  double dx = (xmax-xmin)/(double)nx;
  double dy = (ymax-ymin)/(double)ny;

  double i_max_d;
  
  double max_d_curr = 0.0, i_h_cb;
  int start_ind = 0, end_ind = 0;
  
  int i,j,pind,Nper_kernel,Nper_thread,my_start = 0,my_end=0;
  int left,upper,xpos,ypos;
  double x,y,qt,loc_val,ker_val;


  /*
    ------------------------------
    start the loop through kernels
    ------------------------------
  */
  
  // make sure kmin and kmax are odd
  if (!(kmax % 2)) kmax += 1;
  if (!(kmin % 2)) kmin += 1;
  kmin = (kmin>1) ? kmin : 1;

  for(int k=kmin; k < kmax+2; k+=2) 
    {
      // set up the base kernel
      kernel_distance(kernel,dx,dy,kmax);
 
      /*
      max distance for this kernel
      */
      max_d_curr = dx*floor(k/2.0);
      max_d_curr = (max_d_curr < dx/2.0) ? dx/2.0 : max_d_curr;
      
      i_max_d = 1./max_d_curr;
      
      /* -------------------------------------------------
         find the chunk of particles that need this kernel
         ------------------------------------------------- */
      
      for(end_ind=start_ind;end_ind<Npart;) { 
        if (2*hs[end_ind] < max_d_curr) end_ind++;
        else break;
      }
      Nper_kernel = end_ind-start_ind;


      /*-------------------------------------------------------------------------
        only continue with kernel generation if there are particles that need it!
        -------------------------------------------------------------------------*/
      if (Nper_kernel > 0) 
        {
          kernel_func(kernel,1.0,max_d_curr,kmax);
          i_h_cb = 8.*i_max_d*i_max_d*i_max_d;
          
	  /*
            paint each particle on the image
          */

          for (pind=start_ind;pind<end_ind;pind++)
            {
              x = xs[pind];
              y = ys[pind];
              //h = hs[inds[pind]];
              qt = qts[pind];
              
              xpos = (x-xmin)/dx;
              ypos = (y-ymin)/dy;
              
              left = xpos-k/2;
              upper = ypos-k/2;
              
              for(i = 0; i < k; i++) 
                {
                  for(j = 0; j < k; j++) 
                    {
                      if((i+left >= 0) && (i+left < nx) &&
                         (j+upper >= 0) && (j+upper < ny))
                        {

                          ker_val = kernel[(i+(kmax-k)/2)+kmax*(j+(kmax-k)/2)]*qt*i_h_cb;
                          image[(i+left)+(j+upper)*nx] += ker_val;

                        }
		    }
		}
	    }
	}
      
      start_ind = end_ind;

    } // closees the k for loop
}



