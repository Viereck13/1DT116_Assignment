// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

// #include <stdio.h>
// #include "ped_model.h"

// #include <cstdlib>
// #include <iostream>
// #include <cmath>
// using namespace std;

// // Memory leak check with msvc++
// #include <stdlib.h>

// __global__ void heatmapTickKernel(int* heatmap_d, int* scaled_heatmap_d, int* blurred_heatmap_d, int numAgents, int* desiredX, int* desiredY);

// void heatTickCUDA(int* heatmap_d, int* scaled_heatmap_d, int* blurred_heatmap_d, int numAgents, int* desiredX, int* desiredY) {
//     // Launch the Vector Add CUDA Kernel
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (SCALED_SIZE*SCALED_SIZE + threadsPerBlock - 1) / threadsPerBlock;
//     // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
//     //      threadsPerBlock);
//     heatmapTickKernel<<<blocksPerGrid, threadsPerBlock>>>(
//         heatmap_d, scaled_heatmap_d, blurred_heatmap_d, numAgents, desiredX, desiredY);
// }

// void setupHeatmapCUDA(int* heatmap_d, int* scaled_heatmap_d, int* blurred_heatmap_d)
// {
//     cudaMalloc(&heatmap_d, SIZE*SIZE*sizeof(int));
//     cudaMemset(&heatmap_d, 0, SIZE*SIZE*sizeof(int));
//     cudaMalloc(&scaled_heatmap_d, SCALED_SIZE*SCALED_SIZE*sizeof(int));
//     cudaMalloc(&blurred_heatmap_d, SCALED_SIZE*SCALED_SIZE*sizeof(int));
// }

// void copyBackHeatmapCUDA(int** heatmap, int* heatmap_d, int** scaled_heatmap, int* scaled_heatmap_d, int** blurred_heatmap, int* blurred_heatmap_d)
// {
//     int* heatmap_l;
//     int* scaled_heatmap_l;
//     int* blurred_heatmap_l;

//     cudaMallocHost(&heatmap_l, SIZE*SIZE*sizeof(int));
//     cudaMemset(&heatmap_l, 0, SIZE*SIZE*sizeof(int));
//     cudaMallocHost(&scaled_heatmap_l, SCALED_SIZE*SCALED_SIZE*sizeof(int));
//     cudaMallocHost(&blurred_heatmap_l, SCALED_SIZE*SCALED_SIZE*sizeof(int));

//     cudaMemcpy(&heatmap_l, &heatmap_d,  SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&scaled_heatmap_l, &scaled_heatmap_d, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&blurred_heatmap_l, &blurred_heatmap_d, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

//     for (size_t i = 0; i < SIZE*SIZE; i++)
//     {
//         heatmap[i/SIZE][i%SIZE] = heatmap_l[i];
//     }
    
//     std::cout << "PROBLEM: " << sizeof(scaled_heatmap) << std::endl;
//     for (size_t i = 0; i < 1000*1000; i++)
//     {
//         scaled_heatmap[i/SCALED_SIZE][i%SCALED_SIZE] = scaled_heatmap_l[i];
//         blurred_heatmap[i/SCALED_SIZE][i%SCALED_SIZE] = blurred_heatmap_l[i];
//     }
// }

// __global__ void heatmapTickKernel(int* heatmap_d, int* scaled_heatmap_d, int* blurred_heatmap_d, int numAgents, int* desiredX, int* desiredY) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     // if (i >= numElements) return;
//     if (i < SIZE*SIZE) {
//         heatmap_d[i] = (int)round(heatmap_d[i] * 0.80);
//     }

//     __syncthreads();

//     if (i < numAgents)
//     {
//         int x = desiredX[i];
//         int y = desiredY[i];
    
//         if (x >= 0 || x < SIZE || y >= 0 || y < SIZE)
//         {
//             atomicAdd(&heatmap_d[y*SIZE+x], 40);
//         }
//     }
    
//     __syncthreads();

//     if (i < SIZE*SIZE)
// 	{
//         heatmap_d[i] = heatmap_d[i] < 255 ? heatmap_d[i] : 255;
// 	}
    
//     __syncthreads();

//     // Scale the data for visual representation
//     // __shared__ int scaled_heatmap_d_s[SCALED_SIZE*SCALED_SIZE];
//     if (i < SIZE*SIZE)
//     {
//         int value = heatmap_d[i];
//         for (size_t j = 0; j < CELLSIZE; j++)
//         {
//             scaled_heatmap_d[i*CELLSIZE+j] = value;
//         }
//     }

//     __syncthreads();

//     // Weights for blur filter
//     const int w[5][5] = {
//         { 1, 4, 7, 4, 1 },
//         { 4, 16, 26, 16, 4 },
//         { 7, 26, 41, 26, 7 },
//         { 4, 16, 26, 16, 4 },
//         { 1, 4, 7, 4, 1 }
//     };

//     #define WEIGHTSUM 273
    
//     if (i < SCALED_SIZE*SCALED_SIZE) {
//         int x = i/SCALED_SIZE;
//         int y = i%SCALED_SIZE;
//         if (x < 2 || x >= SCALED_SIZE-2) {
//             return;
//         }
//         if (y < 2 || y >= SCALED_SIZE-2) {
//             return;
//         }

//         int sum = 0;
//         for (int k = -2; k < 3; k++)
//         {
//             for (int l = -2; l < 3; l++)
//             {
//                 sum += w[2 + k][2 + l] * scaled_heatmap_d[(x + k)*SCALED_SIZE + y + l];
//             }
//         }
//         int value = sum / WEIGHTSUM;
//         blurred_heatmap_d[x * SCALED_SIZE + y] = 0x00FF0000 | value << 24;
//     }
// }