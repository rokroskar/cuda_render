#include <stdio.h>

#define KSIZE 31
#define IMAGE_XDIM 100
#define IMAGE_YDIM 100
#define IMAGE_SIZE IMAGE_XDIM*IMAGE_YDIM

#define PI 3.141592654f
#define PI_I 1.0/PI

struct Coords_f {
  float xmin;
  float xmax;
  float ymin;
  float ymax;
};

struct Coords_i {
  int xmin,xmax,ymin,ymax;
};

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
          d = kernel[i]/(max_d/2.0);
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
  if (idx == blockDim.x-1) my_end = nx_loc*ny_loc;
  
  for(int p = my_start; p < my_end; p++) 
    {
      loc_y = p/nx_loc;
      loc_x = p - loc_y*nx_loc;
      global[(loc_x+x_offset) + (loc_y+y_offset)*nx_glob] += local[p];
    }
}


__global__ void tile_render_kernel(float *xs, float *ys, float *qts, float *hs, int *inds, int *p_per_tile,
                                   float xmin, float xmax, float ymin, float ymax, 
                                   float *global_image, int nx_glob, int ny_glob)
{    

  int  Nthreads = blockDim.x;
  int idx = threadIdx.x;
  
  
  // declare shared arrays -- image and base kernel
  __shared__ float local_image[IMAGE_XDIM*IMAGE_YDIM];
  __shared__ float kernel[KSIZE*KSIZE];

  int nx;
  int ny;
  
  float dx,dy; 

  Coords_i tile_pix; 
  Coords_f tile_phy; // pixel and physical limits of the tile processed by the block
  Coords_f global_coords;

  float i_max_d;
  
  float max_d_curr = 0.0, i_h_cb;
  int start_ind = 0, end_ind = 0;
  int kmin=1,kmax=31;
  int i,j,pind,Nper_kernel,Nper_thread,my_start = 0,my_end=0;
  int left,upper,xpos,ypos;
  float x,y,qt,loc_val,ker_val;
  int ind_offset = 0;

  global_coords.xmin = xmin;
  global_coords.xmax = xmax;
  global_coords.ymin = ymin;
  global_coords.ymax = ymax;
  
  dx = (global_coords.xmax-global_coords.xmin)/float(nx_glob);
  dy = (global_coords.ymax-global_coords.ymin)/float(ny_glob);
  /* 
    figure out which part of the image this block is responsible for
  */

  tile_pix.xmin = blockIdx.x*IMAGE_XDIM;
  tile_pix.ymin = blockIdx.y*IMAGE_YDIM;
  tile_pix.xmax = (blockIdx.x < gridDim.x -1) ? (blockIdx.x+1)*IMAGE_XDIM - 1 : nx_glob-1;
  tile_pix.ymax = (blockIdx.y < gridDim.y -1) ? (blockIdx.y+1)*IMAGE_YDIM - 1 : ny_glob-1;
  
  nx = tile_pix.xmax-tile_pix.xmin;
  ny = tile_pix.ymax-tile_pix.ymin;

  tile_phy.xmin = global_coords.xmin + dx*tile_pix.xmin;
  tile_phy.ymin = global_coords.ymin + dy*tile_pix.ymin;
  tile_phy.xmax = global_coords.xmin + dx*(tile_pix.xmax+1);
  tile_phy.ymin = global_coords.ymin + dy*(tile_pix.ymax+1);
  
  
  // set up the index array offset
  for(i=0;i<blockIdx.x;i++) ind_offset += p_per_tile[i];
  start_ind = ind_offset;

  /*
    ------------------------------
    start the loop through kernels
    ------------------------------
  */
  
  // make sure kmin and kmax are odd
  //if (!(kmax % 2)) kmax += 1;
  //if (!(kmin % 2)) kmin += 1;
  //kmin = (kmin>1) ? kmin : 1;

  if(idx < IMAGE_SIZE) for(i=idx;i<IMAGE_SIZE;i+=Nthreads) local_image[i]=0.0;
      
  if (idx==0) {
    printf("block = %d min/max = %f %f\n", blockIdx.x+blockIdx.y*gridDim.y, tile_phy.xmin, tile_phy.xmax);
    printf("pixels xmin/xmax = %d %d\n", tile_pix.xmin, tile_pix.xmax);
    
  }
  //for(int m=0;m<5000;m++){
  for(int k=1; k <= 31; k+=2) 
    {
      __syncthreads();
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
      for(end_ind = start_ind; end_ind < p_per_tile[blockIdx.x+blockIdx.y*gridDim.y]+ind_offset; ) { 
        if (2*hs[inds[end_ind]] < max_d_curr) end_ind++;
        else break;
      }
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
          if (idx == Nthreads-1) 
            my_end = end_ind;
          else 
            my_end = Nper_thread+my_start;
          
          if (idx == 0) printf("k = %d Nperkernel = %d Nper_thread = %d\n", k, Nper_kernel, Nper_thread);
          printf("idx = %d my start = %d my end = %d\n",idx, my_start,my_end);

          //all threads have their particle indices figured out, increment for next iteration
         
          start_ind = end_ind;
          
          /*
            paint each particle on the local image
          */

          for (pind=my_start;pind<my_end;pind++)
            {
              x = xs[inds[pind]];
              y = ys[inds[pind]];
              //h = hs[inds[pind]];
              qt = qts[inds[pind]];
              
              xpos = __float2int_rd((x-tile_phy.xmin)/dx);
              ypos = __float2int_rd((y-tile_phy.ymin)/dy);
              
              left = xpos-k/2;
              upper = ypos-k/2;
              
              for(i = 0; i < k; i++) 
                {
                  for(j = 0; j < k; j++) 
                    {
                      if((i+left >= 0) && (i+left < nx) &&
                         (j+upper >= 0) && (j+upper < ny))
                        {
                          ker_val = kernel[(i+(KSIZE-k)/2)+KSIZE*(j+(KSIZE-k)/2)]*qt*i_h_cb;
                          loc_val = local_image[(i+left)+(j+upper)*nx];
                          local_image[(i+left)+(j+upper)*nx] = loc_val + ker_val;
                        }
                    }
                }
            }
        }
    }
  __syncthreads();
  /* update global image */
  update_image(global_image,local_image,tile_pix.xmin,tile_pix.ymin,nx_glob,nx,ny);
  //  }
}



