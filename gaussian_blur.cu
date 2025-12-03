#include <cuda_runtime.h>
#Define BLUR_SIZE 5

__global__ void blur_kernel(unsigned char* image,unsigned* char blurred,unsigned int width,unsigned int height){
    int outRow = blockIdx.y *blockDim.y +threadIdx.y;
    int outCol =blockIdx.x *blockDim.x +threadIdx.x;
    if(outRow<height && outCol<width){
        unsigned int average =0;
        for(int inRow =outRow -BLUR_SIZE;inRow <outRow+BLUR_SIZE+1,++inRow){
            for(int inCol =outCol -BLUR_SIZE;inCol < outCOl +BLUR_SIZE +1;++inCol){
                if(inRow>=0 && inRow<height && inCol>=0 && inCol<width){
                    average +=image[inRow*width +inCol];
                }
                

            }
        }
        blurred[outRow*width +outCol]=(unsigned char)(average/((2 * BLUR_SIZE+1)*(2 * BLUR_SIZE +1)));
    }
}

void blur_gpu(unsigned char* image ,unsigned char* blurred ,unsigned int width,unsigned char* height){
    unsigned char *image_d ,*blurred_d;
    cudaMalloc((void**) &image_d , width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blurred_d,width*height*sizeof(unsigned_char));
    cudaDeviceSynchronize();

    //Copy data to GPU
    cudaMemcpy(image_d,image,width * height *sizeof(unsigned_char));
    // cudaMemcpy(blurred_d,blurred,width*height*sizeof(unsigned_char));
    cudaDeviceSynchronize();
    //Call Kernel
    dim3 numThreadsPerBlock(16,16);
    dim3 numBlocks((width +numThreadsPerBlock.x-1)/numThreadsPerBlock.x,(height +numThreadsPerBlock.y-1)/numThreadsPerBlock.y);
    cudaDeviceSynchronize();

    //Free the GPU Memory
    cudaFree(image_d);
    cudaFree(blurred_d);
    cudaDeviceSynchronize();

}