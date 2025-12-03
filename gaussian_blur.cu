#include <cuda_runtime.h>
#include <stdio.h>
#Define BLUR_SIZE 5

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        // exit(EXIT_FAILURE); // Uncomment to exit on error
    }
}

__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    // A safer type for average to prevent overflow, especially for larger BLUR_SIZE
    unsigned long long average = 0; 
    
    // Determine the output pixel coordinates
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (outRow < height && outCol < width) {
        int count = 0;
        
        // Loop for input rows in the convolution window
        for (int inRow = outRow - BLUR_SIZE; inRow <= outRow + BLUR_SIZE; ++inRow) {
            // Loop for input columns in the convolution window
            for (int inCol = outCol - BLUR_SIZE; inCol <= outCol + BLUR_SIZE; ++inCol) {
                // Check bounds for the input image (handle edge pixels)
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    average += image[inRow * width + inCol];
                    count++;
                }
            }
        }
        // Division by the full window size: (2*BLUR_SIZE + 1) * (2*BLUR_SIZE + 1)
        int divisor = (2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1); 
        
        // If we want a perfectly averaged blur, divide by 'count'
        if (count > 0) {
            blurred[outRow * width + outCol] = (unsigned char)(average / count);
        } else {
            // Should not happen if the output is within bounds, but as a safeguard
            blurred[outRow * width + outCol] = 0; 
        }
        
        // Standard (simpler) implementation often uses the full divisor:
        // blurred[outRow * width + outCol] = (unsigned char)(average / divisor);
    }
}
// Corrected Host Function
void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    unsigned char *image_d, *blurred_d;
    size_t size = (size_t)width * height * sizeof(unsigned char);

    // 1. Allocate GPU Memory
    checkCudaErrors(cudaMalloc((void**) &image_d, size));
    checkCudaErrors(cudaMalloc((void**) &blurred_d, size));
    
    // Copy data from Host to GPU
    // cudaMemcpy is synchronous by default, so cudaDeviceSynchronize is often not strictly needed here
    checkCudaErrors(cudaMemcpy(image_d, image, size, cudaMemcpyHostToDevice));

    // Setup Grid and Block Dimensions
    dim3 numThreadsPerBlock(16, 16);
    // Integer division ceiling: (a + b - 1) / b
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 
                   (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    // Call Kernel
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);

    cudaDeviceSynchronize();

    // Copy data from GPU to Host
    checkCudaErrors(cudaMemcpy(blurred, blurred_d, size, cudaMemcpyDeviceToHost));
    
    // Free the GPU Memory
    checkCudaErrors(cudaFree(image_d));
    checkCudaErrors(cudaFree(blurred_d));
}
