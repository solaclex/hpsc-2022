#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void count(int *key,int *bucket){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&bucket[key[i]+1], 1);
}

__global__ void scan(int *bucket,int *tmp,int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=1; j<n; j<<=1) {
    tmp[i+1] = bucket[i+1];
    __syncthreads();
    if(i>=j) bucket[i+1] += tmp[i-j+1];
    __syncthreads();
  }
}

__global__ void put(int *key,int *bucket){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j = bucket[i]; j <bucket[i+1];j++){
    key[j] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *tmp;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, (range+1)*sizeof(int));
  cudaMallocManaged(&tmp, (range+1)*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  //std::vector<int> bucket(range); 
  for (int i=0; i<range+1; i++) {
    bucket[i] = 0;
    tmp[i] = 0;
  }

  count<<<1,n>>>(key,bucket);
  cudaDeviceSynchronize();
  scan<<<1,range>>>(bucket,tmp,range);
  cudaDeviceSynchronize();
  put<<<1,range>>>(key,bucket);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
