#include <stdio.h>

#define KSIZE 31
#define IMAGE_XDIM 100
#define IMAGE_YDIM 100
#define IMAGE_SIZE IMAGE_XDIM*IMAGE_YDIM

#define PI 3.141592654f
#define PI_I 1.0/PI

__device__ void kernel_func(float *kernel, float h, float max_d)
{
  /* 
     calculate the kernel value given the distance from center 
     normalized to 2*smoothing_length
  */
  int idx = threadIdx.x,my_start,my_end;
  float d;  
  float pix_per_thread = __int_as_float(KSIZE*KSIZE)/__int_as_float(blockDim.x);

  if (pix_per_thread > 1) 
    {
      my_start = __float2int_rd(pix_per_thread)*idx;
      if (idx == blockDim.x-1)
        my_end = KSIZE*KSIZE-1;
      else
        my_end  = __float2int_rd(pix_per_thread)*(idx+1);
    
      for(int i=my_start;i<my_end;i++) 
        {
          d = kernel[i]/max_d;
          kernel[i] = 0.0;
          if(d < 1) kernel[i] = (1.-(3./2)*(d*d) + (3./4.)*(d*d*d))*PI_I/(h*h*h);
          else if (d <= 2.0) kernel[i] = 0.25*powf((2.-d),3)*PI_I/(h*h*h); 
        }
    }
  else
    {
      if (idx < KSIZE*KSIZE) 
        {
          d = kernel[idx]/(max_d/2.0);
          kernel[idx] = 0.0;
          if(d < 1) kernel[idx] = (1.-(3./2)*(d*d) + (3./4.)*(d*d*d))*PI_I/(h*h*h);
          else if (d <= 2.0) kernel[idx] = 0.25*powf((2.-d),3)*PI_I/(h*h*h);    
        }
    }  
}

__device__ void kernel_distance(float *kernel,float dx,float dy) 
{
  /*
    calculate distance of pixel from center in physical units
    each thread processes one column
  */

  int idx = threadIdx.x;
  int cen;
  int x,y,my_start,my_end;
  float dxpix,dypix;
  float pix_per_thread = __int_as_float(KSIZE*KSIZE)/__int_as_float(blockDim.x);
  cen = KSIZE/2;
  
  if(pix_per_thread > 1)
    {
      my_start = __float2int_rd(pix_per_thread)*idx;
      if (idx == blockDim.x-1)
        my_end = KSIZE*KSIZE-1;
      else
        my_end  = __float2int_rd(pix_per_thread)*(idx+1);

      for(int i=my_start;i<my_end;i++) 
        {
          x = i/KSIZE;
          y = i - x*KSIZE;

          dxpix = float(x-cen)*dx;
          dypix = float(y-cen)*dy;
          kernel[y*KSIZE+x] = sqrtf(dxpix*dxpix + dypix*dypix);
        }
    }
  else
    {      
      if (idx < KSIZE*KSIZE) 
        {
          x = idx/KSIZE;
          y = idx - x*KSIZE;

          dxpix = float(x-cen)*dx;
          dypix = float(y-cen)*dy;
          kernel[y*KSIZE+x] = sqrtf(dxpix*dxpix + dypix*dypix);
        }
    }
}


__device__ void update_image(float *global, float *local, int x_offset, int y_offset, int nx_glob, int nx_loc, int ny_loc)
{
  int idx = threadIdx.x;
  int pix_per_thread = nx_loc*ny_loc/(blockDim.x);

  int my_start, my_end, loc_x, loc_y;
  
  my_start = idx*pix_per_thread;
  my_end = my_start + pix_per_thread;
  // if this is the last thread, make it take the rest of the pixels
  if (blockIdx.x == blockDim.x-1) my_end = nx_loc*ny_loc;
  
  for(int p = my_start; p < my_end; p++) 
    {
      loc_x = p/nx_loc;
      loc_y = p - loc_x*nx_loc;
      global[nx_glob*(loc_x+x_offset) + loc_y+y_offset] += local[p];
    }
}

__global__ void tile_render_kernel(float *xs, float *ys, float *qts, float *hs, int *inds, int Npart,
                                   int kmin, int kmax, 
                                   float xmin_p, float xmax_p, float ymin_p, float ymax_p,
                                   int xmin, int xmax, int ymin, int ymax, 
                                   float *global_image, int nx_glob, int ny_glob)
{    

  int  Nthreads = blockDim.x;
  int idx = threadIdx.x;

  
  // declare shared arrays -- image and base kernel
  __shared__ float local_image[IMAGE_XDIM*IMAGE_YDIM];
  __shared__ float kernel[KSIZE*KSIZE];

  int nx = xmax-xmin+1;
  int ny = ymax-ymin+1;
  
  float dx = (xmax_p-xmin_p)/float(nx);
  float dy = (ymax_p-ymin_p)/float(ny);

  float i_max_d;
  
  float max_d_curr = 0.0, i_h_cb;
  int start_ind = 0, end_ind = 0;
  
  int i,j,pind,Nper_kernel,Nper_thread,my_start = 0,my_end=0;
  int left,upper,xpos,ypos;
  float x,y,qt,loc_val,ker_val;


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
      kernel_distance(kernel,dx,dy);
 
      /*
      max distance for this kernel
      */
      max_d_curr = dx*floorf(k/2.0);
      max_d_curr = (max_d_curr < dx/2.0) ? dx/2.0 : max_d_curr;
      
      i_max_d = 1./max_d_curr;
      
      /* -------------------------------------------------
         find the chunk of particles that need this kernel
         ------------------------------------------------- */
      
      /* DO THIS SEARCH IN PARALLEL */

      for(j=start_ind;j<Npart;j++) { 
        if (2*hs[j] < max_d_curr) continue;
        else break;
      }
        end_ind = j;
        
        Nper_kernel = end_ind-start_ind;

        
        /*-------------------------------------------------------------------------
          only continue with kernel generation if there are particles that need it!
          -------------------------------------------------------------------------*/
        if (Nper_kernel > 0) 
          {
            kernel_func(kernel,1.0,max_d_curr);
            i_h_cb = 8.*i_max_d*i_max_d*i_max_d;
            
            /* --------------------------------------
               determine thread particle distribution
               --------------------------------------*/
            Nper_thread = Nper_kernel/Nthreads;
            my_start = Nper_thread*idx+start_ind;
        
            // if this is the last thread, make it pick up the slack
            my_end = end_ind;
            if (idx == Nthreads-1) 
              my_end = end_ind;
            else 
              my_end = Nper_thread+my_start;
                    
            //all threads have their particle indices figured out, increment for next iteration
            start_ind = end_ind;
            
            /*
              paint each particle on the local image
            */
            __syncthreads();
            
            for (pind=my_start;pind<my_end;pind++)
              {
                x = xs[inds[pind]];
                y = ys[inds[pind]];
                //h = hs[inds[pind]];
                qt = qts[inds[pind]];
                
                xpos = __float2int_rd((x-xmin_p)/dx);
                ypos = __float2int_rd((y-ymin_p)/dy);
                
                left = xpos-k/2;
                upper = ypos-k/2;
                if (k==3) printf("%f %f\n", x, y);
                for(i = (KSIZE-k)/2; i < KSIZE-(KSIZE-k)/2; i++) 
                  {
                    for(j = (KSIZE-k)/2; j < KSIZE -(KSIZE-k)/2; j++) 
                      {
                        if((i+left > 0) && (i+left < nx) &&
                           (j+upper > 0) && (j+upper < ny))
                          {
                            ker_val = kernel[i+KSIZE*j]*qt*i_h_cb;
                            loc_val = local_image[(i+left)+(j+upper)*nx];
                            local_image[(i+left)+(j+upper)*nx] = loc_val + ker_val;
                            if (k==3) printf("%d %d %d %d %f  ",i,j,i+left,j+upper,ker_val);
                          }
                      }
                    if(k==3)printf("\n");
                  }
              }
          }
        if (end_ind == Npart-1) break;
    }
  __syncthreads();
  /* update global image */
  //update_image(global_image,local_image,xmin,ymin,nx_glob,nx,ny);
  if (idx == 0) 
    {
      for(i=0;i<100*100;i++) global_image[i] = local_image[i];
    }
    
}



