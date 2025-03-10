#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE
#define WEIGHTSUM 273


// Kernel 1: Fade heatmap – multiply all of previous pixels by 20%.
__global__ void fadeHeatmapKernel(int* d_heatmap, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<totalPixels) {
        d_heatmap[idx] = (int)(d_heatmap[idx]*0.80f+0.5f); // 0.5f is for rounding. 8.1+0.5=8.6 -> 8, but 8.6+0.5=9
        // printf("d_heatmap[%d] = %d\n", idx, d_heatmap[idx]);
    }
}


// Kernel 2: Add color intensity to the desired position using atomicAdd
__global__ void addAgentHeatKernel(int* d_heatmap, int size,
                                   const int* d_agentDesiredX,
                                   const int* d_agentDesiredY,
                                   int numAgents) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < numAgents) {
        int x = d_agentDesiredX[idx];
        int y = d_agentDesiredY[idx];
        // printf("Agent %d at (%d, %d)\n", idx, x, y);
        if (x >= 0 && x < size && y >= 0 && y < size) {
            atomicAdd(&d_heatmap[y*size+x], 40);
            // printf("Heatmap[%d][%d] = %d\n", x, y, d_heatmap[y * size + x]);
        }
    }
}

// Kernel 3: Limit heatmap values to 255
__global__ void limitHeatmapValueKernel(int* d_heatmap, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalPixels && d_heatmap[idx] > 255) {
        d_heatmap[idx]=255;
    }
}

// Kernel 4: Scale heatmap
__global__ void scaleHeatmapKernel(const int* d_heatmap, int* d_scaledHeatmap,
                                   int size, int cellSize) {
    int scaledSize = size*cellSize;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < scaledSize && y < scaledSize) {
        int origX = x/cellSize;
        int origY = y/cellSize;
        d_scaledHeatmap[y * scaledSize + x] = d_heatmap[origY * size + origX];
        // printf("Scaled Heatmap[%d][%d] = %d\n", y, x, d_scaledHeatmap[y * scaledSize + x]);
    }
}

// Kernel 5: Blur filter using shared memory and a 5x5 Gaussian kernel.
// Each thread computes one output pixel (except near the borders).
__global__ void blurFilterKernel(const int* d_scaledHeatmap, int* d_blurredHeatmap, int scaledSize) {
    // Allocate shared memory: tile dimensions plus a 2-pixel halo (aura) on each side.
    extern __shared__ int tile[]; // Know its size at runtime from <<<..., ..., sharedMemSize, ...>>>
    int tileWidth = blockDim.x + 4; // extra columns for halo

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int globalX = blockIdx.x * blockDim.x + tx;
    int globalY = blockIdx.y * blockDim.y + ty;
    int sharedX = tx+2;
    int sharedY = ty+2;

    // Load central data.
    if (globalX < scaledSize && globalY < scaledSize) {
        tile[sharedY*tileWidth + sharedX] = d_scaledHeatmap[globalY*scaledSize + globalX];
    } else {
        tile[sharedY*tileWidth + sharedX] = 0;
    }

    // Load halo for left and right edges.
    // Get two pixels to the left of the block.
    if (tx<2) {
        int gx = globalX-2;
        if (gx >= 0 && globalY < scaledSize)
            tile[sharedY*tileWidth+tx] = d_scaledHeatmap[globalY*scaledSize+gx];
        else
            tile[sharedY*tileWidth+tx] = 0;
    }
    // Get two pixels to the right of the block.
    if (tx >= blockDim.x-2) {
        int gx = globalX+2;
        if (gx < scaledSize && globalY < scaledSize)
            tile[sharedY * tileWidth + sharedX + 2] = d_scaledHeatmap[globalY * scaledSize + gx];
        else
            tile[sharedY * tileWidth + sharedX + 2] = 0;
    }

    // Load halo data for top and bottom edges.
    // Get two pixels above the block.
    if (ty<2) {
        int gy = globalY-2;
        if (gy >= 0 && globalX < scaledSize)
            tile[ty*tileWidth + sharedX] = d_scaledHeatmap[gy*scaledSize + globalX];
        else
            tile[ty*tileWidth + sharedX] = 0;
    }
    // Get two pixels below the block.
    if (ty >= blockDim.y-2) {
        int gy = globalY+2;
        if (gy < scaledSize && globalX < scaledSize)
            tile[(sharedY+2) * tileWidth + sharedX] = d_scaledHeatmap[gy*scaledSize + globalX];
        else
            tile[(sharedY+2) * tileWidth + sharedX] = 0;
    }

    // Load corner halo data.
    // Top-left corner.
    if (tx<2 && ty<2) {
        int gx = globalX-2;
        int gy = globalY-2;
        if (gx >= 0 && gy >= 0)
            tile[ty*tileWidth+tx] = d_scaledHeatmap[gy*scaledSize+gx];
        else
            tile[ty*tileWidth+tx] = 0;
    }
    // Top-right corner.
    if (tx >= blockDim.x-2 && ty < 2) {
        int gx = globalX+2;
        int gy = globalY-2;
        if (gx < scaledSize && gy >= 0)
            tile[ty*tileWidth + sharedX +2] = d_scaledHeatmap[gy*scaledSize + gx];
        else
            tile[ty*tileWidth + sharedX +2] = 0;
    }
    // Bottom-left corner.
    if (tx < 2 && ty >= blockDim.y-2) {
        int gx = globalX-2;
        int gy = globalY+2;
        if (gx >= 0 && gy < scaledSize)
            tile[(sharedY+2) * tileWidth +tx] = d_scaledHeatmap[gy * scaledSize +gx];
        else
            tile[(sharedY+2) * tileWidth +tx] = 0;
    }
    // Bottom-right corner.
    if (tx >= blockDim.x-2 && ty >= blockDim.y-2) {
        int gx = globalX+2;
        int gy = globalY+2;
        if (gx < scaledSize && gy < scaledSize)
            tile[(sharedY+2) * tileWidth + sharedX+2] = d_scaledHeatmap[gy*scaledSize + gx];
        else
            tile[(sharedY+2) * tileWidth + sharedX+2] = 0;
    }
    __syncthreads();

    // Only process if within valid bounds (No index out of bound).
    if (globalX >= 2 && globalX < scaledSize-2 && globalY >= 2 && globalY < scaledSize-2) {
        int weights[5][5] = {
            { 1, 4, 7, 4, 1 },
            { 4, 16, 26, 16, 4 },
            { 7, 26, 41, 26, 7 },
            { 4, 16, 26, 16, 4 },
            { 1, 4, 7, 4, 1 }
        };
        int sum = 0;
        for (int ky=-2; ky<=2; ky++) {
            for (int kx=-2; kx<=2; kx++) {
                sum += weights[ky+2][kx+2] * tile[(sharedY+ky)*tileWidth + (sharedX+kx)];
            }
        }
        int value = sum / WEIGHTSUM;
        d_blurredHeatmap[globalY*scaledSize+globalX] = 0x00FF0000 | (value << 24);
    }
}

void updateHeatmapCUDAAsync(int* h_heatmap, int* h_scaledHeatmap, int* h_blurredHeatmap,
    const int* h_agentDesiredX, const int* h_agentDesiredY, int numAgents,
    cudaStream_t stream)
{
    int totalPixels = SIZE * SIZE;
    size_t heatmapSizeBytes = totalPixels * sizeof(int);
    size_t scaledTotalPixels = SCALED_SIZE * SCALED_SIZE;
    size_t scaledSizeBytes = scaledTotalPixels * sizeof(int);

    // Device pointers
    int *d_heatmap = nullptr;
    int *d_scaledHeatmap = nullptr;
    int *d_blurredHeatmap = nullptr;
    int *d_agentDesiredX = nullptr;
    int *d_agentDesiredY = nullptr;

    cudaMalloc((void**)&d_heatmap, heatmapSizeBytes);
    cudaMalloc((void**)&d_scaledHeatmap, scaledSizeBytes);
    cudaMalloc((void**)&d_blurredHeatmap, scaledSizeBytes);
    cudaMalloc((void**)&d_agentDesiredX, numAgents * sizeof(int));
    cudaMalloc((void**)&d_agentDesiredY, numAgents * sizeof(int));


    // Copy initial heatmap to device memory asynchronously.
    cudaMemcpyAsync(d_heatmap, h_heatmap, heatmapSizeBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_agentDesiredX, h_agentDesiredX, numAgents * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_agentDesiredY, h_agentDesiredY, numAgents * sizeof(int), cudaMemcpyHostToDevice, stream);
    // cudaMemcpy(d_heatmap, h_heatmap, heatmapSizeBytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_agentDesiredX, h_agentDesiredX, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_agentDesiredY, h_agentDesiredY, numAgents * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; // divisible by 32 (warp size)
    int blocksForFade = (totalPixels+threadsPerBlock-1) / threadsPerBlock;
    int blocksForAgents = (numAgents+threadsPerBlock-1) / threadsPerBlock;
    dim3 blockDim2D(16, 16); 
    dim3 gridDim2D((SCALED_SIZE+blockDim2D.x -1) / blockDim2D.x,
    (SCALED_SIZE+blockDim2D.y -1) / blockDim2D.y);
    // determine the number of grids by SCALED_SIZE/blockDim2D.x and SCALED_SIZE/blockDim2D.y
    // (SCALED_SIZE + blockDim2D.x - 1) / blockDim2D.x to allow for partial blocks
    size_t sharedMemSize = (blockDim2D.x + 4) * (blockDim2D.y + 4) * sizeof(int); // +4 for halo, 2 on each side

    // Multiple kernels added to the stream will be executed in order of their addition.
    // No need of cudaDeviceSynchronize() after each kernel launch.

    // Launch Kernel 1
    fadeHeatmapKernel<<<blocksForFade, threadsPerBlock, 0, stream>>>(d_heatmap, totalPixels);

    // Launch Kernel 2
    addAgentHeatKernel<<<blocksForAgents, threadsPerBlock, 0, stream>>>(d_heatmap, SIZE, d_agentDesiredX, d_agentDesiredY, numAgents);

    // Launch Kernel 3
    limitHeatmapValueKernel<<<blocksForFade, threadsPerBlock, 0, stream>>>(d_heatmap, totalPixels);

    // Launch Kernel 4
    scaleHeatmapKernel<<<gridDim2D, blockDim2D, 0, stream>>>(d_heatmap, d_scaledHeatmap, SIZE, CELLSIZE);
    // dim3 blockDim2D(16, 16) tells CUDA that each block should have 16 threads along the x-dimension, 
    // 16 threads along the y-dimension, and 1 thread along the z-dimension
    // resulting in a total of 256 threads per block. These values are used
    // inside the kernel to determine each thread’s unique indices via threadIdx.x, threadIdx.y, and threadIdx.z.

    // Launch Kernel 5
    blurFilterKernel<<<gridDim2D, blockDim2D, sharedMemSize, stream>>>(d_scaledHeatmap, d_blurredHeatmap, SCALED_SIZE);

    // // Launch Artificial Workload Kernel
    // int numElements = 1024;  // You can adjust this size if needed.
    // int *d_dummyData = nullptr;
    // cudaMalloc(&d_dummyData, numElements * sizeof(int));
    // int iterations = 1000000; // Adjust this number to increase the workload.
    // int blocks = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    // artificialWorkloadKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_dummyData, iterations);

    // Copy the final heatmap and blurred heatmap back to host memory asynchronously.
    cudaMemcpyAsync(h_blurredHeatmap, d_blurredHeatmap, scaledSizeBytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_heatmap, d_heatmap, heatmapSizeBytes, cudaMemcpyDeviceToHost, stream);
    // cudaMemcpy(h_blurredHeatmap, d_blurredHeatmap, scaledSizeBytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_heatmap, d_heatmap, heatmapSizeBytes, cudaMemcpyDeviceToHost);

    // Free device memory.
    // cudaStreamSynchronize(stream); // CPU waits for GPU to finish before CPU moves on to the next step.
    // printf("---------Async kernel execution complete-----------------\n");
    cudaFree(d_heatmap);
    cudaFree(d_scaledHeatmap);
    cudaFree(d_blurredHeatmap);
    cudaFree(d_agentDesiredX);
    cudaFree(d_agentDesiredY);
}