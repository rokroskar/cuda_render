#include <stdio.h>

#define KSIZE 101
#define TILE_XDIM 100
#define TILE_YDIM 100
#define TILE_SIZE TILE_XDIM*TILE_YDIM

#define PI 3.141592654f
#define PI_I 1.0/PI

struct Particle {
  float x;
  float y; 
  float qt; 
  float h;
} ;




inline __device__ uint scan1Inclusive(int idata, volatile int *s_Data, int size)
{
    int pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (int offset = 1; offset < size; offset <<= 1)
    {
        __syncthreads();
        int t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(int idata, volatile int *s_Data, int size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}

inline __device__ float kernel_value(float d, float h) 
{
  if (d < 1) return (1.-(3./2)*(d*d) + (3./4.)*(d*d*d))*PI_I/(h*h*h);
  else if (d <= 2.0) return 0.25*powf((2.-d),3)*PI_I/(h*h*h); 
  else return 0.0;
}

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
  int loc_x, loc_y;
  
  for(int p = idx; p < nx_loc*ny_loc; p+=blockDim.x) 
    {
      loc_y = p/nx_loc;
      loc_x = p - loc_y*nx_loc;
      atomicAdd(&global[(loc_x+x_offset) + (loc_y+y_offset)*nx_glob], local[p]);
    }
}

__global__ void tile_render_kernel(Particle *ps, int *tile_offsets, int tile_id, 
                                   float xmin_p, float xmax_p, float ymin_p, float ymax_p,
                                   int xmin, int xmax, int ymin, int ymax, 
                                   float *global_image, int nx_glob, int ny_glob)
{    

  int Nthreads = blockDim.x*gridDim.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int Npart = tile_offsets[tile_id+1] - tile_offsets[tile_id];
  
  // declare shared array for local image 
  __shared__ float local_image[TILE_XDIM*TILE_YDIM];
  __shared__ int counter;
  __shared__ float timer;
  int nx = xmax-xmin+1;
  int ny = ymax-ymin+1;
  
  float dx = (xmax_p-xmin_p)/float(nx);
  float dy = (ymax_p-ymin_p)/float(ny);

  float i_max_d;
  
  float max_d_curr = 0.0, i_h_cb, i_h, d, h;
  int start_ind, end_ind;
  
  int i,j,pind,Nper_kernel;
  int left,upper,xpos,ypos,kmax,kmin;
  float x,y,qt;

  clock_t start_t, end_t;

  start_ind = tile_offsets[tile_id];
  kmin = 1;
  kmax = 31;

  counter =0;
  timer = 0.0;
  /*
    ------------------------------
    start the loop through kernels
    ------------------------------
  */

  for(i=threadIdx.x;i<TILE_SIZE;i+=blockDim.x) local_image[i]=0.0;


  for(int k=kmin; k < kmax+2; k+=2) 
    {
      __syncthreads();
      // set up the base kernel
      //      kernel_distance(kernel,dx,dy);
 
      /*
      max distance for this k
      */
      max_d_curr = dx*floorf(k/2.0);
      max_d_curr = (max_d_curr < dx/2.0) ? dx/2.0 : max_d_curr;
      
      i_max_d = 1./max_d_curr;
      
      /* -------------------------------------------------
         find the chunk of particles that need this kernel
         ------------------------------------------------- */
      
      /* DO THIS SEARCH IN PARALLEL */
      start_t = clock();
      for(end_ind=start_ind;end_ind<tile_offsets[tile_id+1];) { 
        if (2*ps[end_ind].h < max_d_curr) end_ind++;
        else break;
      }
      end_t = clock();
      atomicAdd(&timer,end_t-start_t);

      Nper_kernel = end_ind-start_ind;

      /*-------------------------------------------------------------------------
        only continue with kernel generation if there are particles that need it!
        -------------------------------------------------------------------------*/
      if (Nper_kernel > 0) 
        {
          //  kernel_func(kernel,1.0,max_d_curr);
          i_h_cb = 8.*i_max_d*i_max_d*i_max_d;
          h = max_d_curr/2.0;
          i_h = 1./h;

          /*
            paint each particle on the local image
          */
          counter = 0;
          for (pind=start_ind+idx;pind<end_ind;pind+=Nthreads)
            {
              x =  ps[pind].x;
              y =  ps[pind].y;
              h =  ps[pind].h;
              qt = ps[pind].qt;

              xpos = __float2int_rd((x-xmin_p)/dx);
              ypos = __float2int_rd((y-ymin_p)/dy);
              
              left = xpos-k/2;
              upper = ypos-k/2;
              
              for(i = 0; i < k; i++) 
                {
                  for(j = 0; j < k; j++) 
                    {
                      if((i+left >= 0) && (i+left < nx) &&
                         (j+upper >= 0) && (j+upper < ny))
                        {
                          d = sqrtf((float)(i-k/2)*(i-k/2)*dx*dx+
                                    (float)(j-k/2)*(j-k/2)*dy*dy);

                          atomicAdd(&local_image[(i+left)+(j+upper)*nx],kernel_value(d*i_h, 1.0)*qt*i_h_cb);
                        }
                    }
                }
              atomicAdd(&counter,1);
            }
          start_ind = end_ind;
        }
    }
  __syncthreads();
  /* update global image */
  update_image(global_image,local_image,xmin,ymin,nx_glob,nx,ny);
  if(threadIdx.x==0) printf("tile = %d time spent in search = %f\n", tile_id, timer/(1215.*1000.));
}


__device__ int get_tile_id(float x, float y, float xmin, float xmax, float ymin, float ymax, 
                           int ny, float dx, float dy)
{
  if ((x < xmin) || (y < ymin) || (x > xmax) || (y > ymax)) return -100; // if out of the image
  else return __float2int_rd((x - xmin)/dx)*ny + __float2int_rd((y - ymin)/dy);
}

__device__ int already_marked(int tile_vals[9], int end)
{
  for(int i=0;i<end;i++) 
    if(tile_vals[i] == tile_vals[end]) return 1;
  return 0;
}


__global__ void tile_histogram(Particle *ps, int *hist, int Npart,
                               float xmin, float xmax, float ymin, float ymax,
                               int nx, int ny, int Ntiles)
{

   __shared__ int temp_hist[2500]; 
  int i,j;
  int Nper_thread = Npart/(blockDim.x*gridDim.x);
  int stride = blockDim.x*gridDim.x;
  int tile_vals[9];
  float x,y,h,dx,dy;
  int val;

  dx = (xmax-xmin)/sqrtf(Ntiles);
  dy = (ymax-ymin)/sqrtf(Ntiles);

  for(j = threadIdx.x; j < 2500 ; j += blockDim.x) temp_hist[j] = 0.0;
  
  for(i=threadIdx.x + blockIdx.x*blockDim.x;i < Npart;i+=stride) 
    {
      x = ps[i].x;
      y = ps[i].y;
      h = ps[i].h;

      // center, left, right, up, down, upper left, upper right, lower left, lower right
      float x_offsets[] = {0.,-2*h,2*h,   0.,  0.,-2*h, 2*h, -2*h,  2*h};
      float y_offsets[] = {0.,   0., 0.,2*h,-2*h,  2*h, 2*h, -2*h, -2*h};
      
      for(j=0;j<9;j++) tile_vals[j] = -100;
      
      for(j=0;j<9;j++) 
        {
          tile_vals[j] = get_tile_id(x+x_offsets[j],y+y_offsets[j],xmin,xmax,ymin,ymax,
                                     float2int(sqrtf(Ntiles)),dx,dy);

          if((tile_vals[j] >= 0) && !(already_marked(tile_vals,j)))
            {
              val = tile_vals[j];
              atomicAdd(&(temp_hist[tile_vals[j]]),1);
            }
        }
    }

  __syncthreads();
  
  // copy the local hist to the global hist
  for(i=threadIdx.x;i < Ntiles;i+=blockDim.x) atomicAdd(&(hist[i]),temp_hist[i]);
 
}
  
  
__global__ void distribute_particles(Particle *ps, Particle *ps_tiles, int *tile_offsets, int Npart,
                               float xmin, float xmax, float ymin, float ymax,
                               int nx, int ny, int Ntiles)
{
  extern __shared__ int shared[];
  int *flag = &shared[0];
  int *counter = &shared[blockDim.x*2];

  float x,y,h,qt,dx,dy;
  
  int idx = threadIdx.x;
  int done = 0, i,j;
  int ind, tile_val;
  int offset, my_flag;

  dx = (xmax-xmin)/sqrtf(Ntiles);
  dy = (ymax-ymin)/sqrtf(Ntiles);

  *counter = 0;

  for(uint i=idx;i<Npart;i+=blockDim.x) // each block processes all the particles
    {
      x = ps[i].x;
      y = ps[i].y;
      h = ps[i].h;
      qt = ps[i].qt;

      float x_offsets[] = {0.,-2*h,2*h,0.,0.,-2*h,2*h,-2*h,2*h};
      float y_offsets[] = {0.,0.,0.,2*h,-2*h,2*h,2*h,-2*h,-2*h};
      
      my_flag = 0;

      for(j=0;(j<9) && !my_flag;j++) 
        {
          tile_val = get_tile_id(x+x_offsets[j],y+y_offsets[j],xmin,xmax,ymin,ymax,
                                 float2int(sqrtf(Ntiles)),dx,dy);
          if((tile_val == blockIdx.x) && (my_flag == 0))  
            {
              my_flag = 1;
            }
        }
      
      __syncthreads();
      
      // determine offsets
      offset = scan1Exclusive(my_flag,flag,blockDim.x);
      
      __syncthreads();

      // if the particle the thread read fits into this block, copy it over
      if (my_flag) 
        {
          ind = *counter + offset + tile_offsets[blockIdx.x];
          ps_tiles[ind].x = x;
          ps_tiles[ind].y = y;
          ps_tiles[ind].h = h;
          ps_tiles[ind].qt = qt;
        }
      __syncthreads();
      if (my_flag) atomicAdd(counter,1);
    }
  __syncthreads();
  //  if (idx==0) printf("Block %d had %d particles\n", blockIdx.x,*counter);
}


__global__ void calculate_keys(Particle *ps, int *keys, int Npart, float dx) 
{
  float idx2 = 1./(dx/2.0);
  for(int i=threadIdx.x + blockDim.x*blockIdx.x; i < Npart; i += blockDim.x*gridDim.x)
    keys[i] = 2.0*ps[i].h*idx2;
}


__global__ void tile_render_kernel_single(Particle *ps, int Npart, int *k_offsets,
                                          float xmin_p, float xmax_p, float ymin_p, float ymax_p,
                                          int xmin, int xmax, int ymin, int ymax, 
                                          float *global_image, int nx_glob, int ny_glob)
{    
  __shared__ float kernel[101*101];

  int Nthreads = blockDim.x*gridDim.x;
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  
  int nx = xmax-xmin+1;
  int ny = ymax-ymin+1;
  
  float dx = (xmax_p-xmin_p)/float(nx);
  float dy = (ymax_p-ymin_p)/float(ny);

  float i_max_d;
  
  float max_d_curr = 0.0, i_h_cb;
  int start_ind, end_ind;
  
  int i,j,pind,Nper_kernel,Nper_thread,my_start = 0,my_end=0;
  int left,upper,xpos,ypos,kmax=101,kmin=1;
  float x,y,qt,loc_val,ker_val;
  /*
    ------------------------------
    start the loop through kernels
    ------------------------------
  */

  for(i=idx;i<nx_glob*ny_glob;i+=Nthreads) global_image[i]=0.0;

  for(int k=kmin; k < kmax+2; k+=2) 
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
      
      /*-------------------------------------------------------------------------
        only continue with kernel generation if there are particles that need it!
        -------------------------------------------------------------------------*/
      Nper_kernel = k_offsets[k] - k_offsets[k-1];
 
      if (Nper_kernel > 0) 
        {
          kernel_func(kernel,1.0,max_d_curr);
          i_h_cb = 8.*i_max_d*i_max_d*i_max_d;
          
          /*
            paint each particle on the local image
          */

          for (pind=idx+k_offsets[k-1]; pind < k_offsets[k]; pind += Nthreads)
            {
              x = ps[pind].x;
              y = ps[pind].y;
              //h = hs[inds[pind]];
              qt = ps[pind].qt;
              
              xpos = __float2int_rd((x-xmin_p)/dx);
              ypos = __float2int_rd((y-ymin_p)/dy);
              
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
                          atomicAdd(&global_image[(i+left)+(j+upper)*nx], ker_val);
                        }
                    }
                }
            }
        }
    }
  __syncthreads();
}
