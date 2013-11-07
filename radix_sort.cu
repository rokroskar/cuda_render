#include <stdio.h>
#include <assert.h>
#include <cub/cub.cuh>
#include "radix_sort.h"


using namespace cub;

float radix_sort(int *keys, Particle *ps, int offset, int num_items)
{ 
  /* Note that keys and ps should be device pointers! */

  cudaError_t err; 
  cudaEvent_t start, end;
  int *keys_alt;
  Particle *ps_alt;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // allocate key and value double buffers on the device
  err = cudaMalloc((void**) &keys_alt, num_items*sizeof(int));
  assert(err==0);

  err = cudaMalloc((void**) &ps_alt, num_items*sizeof(Particle));
  assert(err==0);
  
  //printf("offset = %d\n",offset);

  cub::DoubleBuffer<int> d_keys(keys+offset, keys_alt);
  cub::DoubleBuffer<Particle> d_vals(ps+offset, ps_alt);

  // Determine temporary device storage requirements for sorting operation
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items);
  // Allocate temporary storage for sorting operation
  err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
  assert(err==0);
 
  // Run sorting operation
  cudaEventRecord(start,0);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items);
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaDeviceSynchronize();

  cudaEventElapsedTime(&elapsedTime,start,end);
  //  printf("Sort time on GPU = %f ms, %f million keys/s\n", elapsedTime, (float)num_items/elapsedTime*1e3/1e6);

  //Sorted keys are referenced by d_keys.Current()
  keys = d_keys.Current();
  ps = d_vals.Current();
    
  
  cudaFree(keys_alt);
  cudaFree(ps_alt);
  cudaFree(d_temp_storage);
  return elapsedTime;
}
