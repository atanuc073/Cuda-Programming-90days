#include <stdio.h>
#include <stdlib.h>

__global__ void rgb2gray_kernel(unsigned char *red ,unsigned char *green ,unsigned char *blue,unsigned char *gray, unsigned int width , unsigned int height){
    unsigned int row =blockIdx.y * blockDim.y +threadIdx.y;
    unsigned int col =blockIdx.x * blockDim.x +threadIdx.x;

    // chek for boundary
    if(row<height && col <width){
        unsigned int i = row *width +col;
        gray[i] = (red[i] *0.299f) + (green[i] * 0.587f) + (blue[i]* 0.299f);

    }
}


void rgb2gray_gpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray,unsigned int width,unsigned int height){
    // Allocate Memory on Device
    unsigned char *red_d, *green_d, *blue_d, * gray_d;
    cudaMalloc((void**)&red_d,width*height *sizeof(unsigned char));
    cudaMalloc((void**)&green_d,width*height *sizeof(unsigned char));
    cudaMalloc((void**)&blue_d,width*height *sizeof(unsigned char));
    cudaMalloc((void**)&gray_d,width*height *sizeof(unsigned char));
    cudaDeviceSynchronize();

    // Copy data from Host to GPU
    cudaMemcpy(red_d,red,width*height*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(green_d,green,width*height*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d,blue,width * height * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // call Kernel
    dim3 numThreadsPerBlock(32,32,1); // 1024 threads per block
    dim3 numBlocks((width+numThreadsPerBlock.x -1)/numThreadsPerBlock.x,(height+numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    rgb2gray_kernel<<<numBlocks,numThreadsPerBlock>>>(red_d,green_d,blue_d,gray_d,width,height);

}