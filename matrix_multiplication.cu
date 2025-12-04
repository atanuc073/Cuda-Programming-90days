#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void mm_kernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) {
        float value = 0;
        for(int i = 0; i < N; i += 1) {

            size_t idx_a = (size_t)row * N + i;
            size_t idx_b = (size_t)i * N + col;
            value += A[idx_a] * B[idx_b];
        }
        
        C[(size_t)row * N + col] = value;
    }
}

void mm_gpu(float *A, float *B, float *C, int N) {
    float *A_d = NULL, *B_d = NULL, *C_d = NULL;
    size_t size = (size_t)N * N * sizeof(float);

    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 
                   (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);


    cudaError_t err;
    
    err = cudaMalloc((void **) &A_d, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc A failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void **) &B_d, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc B failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void **) &C_d, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc C failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy to the GPU
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    mm_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Copy To the CPU
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

cleanup:
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


// UNIT TEST
void mm_cpu(float *A, float *B, float *C, int N) {
    for(int row = 0; row < N; row++) {
        for(int col = 0; col < N; col++) {
            float value = 0;
            for(int k = 0; k < N; k++) {
                value += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }
}

int main() {
    int N = 64; 
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);

    srand(1337);
    for(int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    mm_cpu(h_A, h_B, h_C_cpu, N);
    mm_gpu(h_A, h_B, h_C_gpu, N);

    float epsilon = 1e-3;
    bool passed = true;
    for(int i = 0; i < N * N; i++) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > epsilon) {
            printf("Mismatch at %d: CPU %f vs GPU %f\n", i, h_C_cpu[i], h_C_gpu[i]);
            passed = false;
            break;
        }
    }

    if (passed) printf("[SUCCESS] Test PASSED! GPU result matches CPU.\n");
    else printf("[FAILURE] Test FAILED!\n");

    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    return 0;
}
