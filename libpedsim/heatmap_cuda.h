#ifndef HEATMAP_CUDA_H
#define HEATMAP_CUDA_H

void updateHeatmapCUDAAsync(int* h_heatmap, int* h_scaledHeatmap, int* h_blurredHeatmap,
                            const int* h_agentDesiredX, const int* h_agentDesiredY,int numAgents, 
                            cudaStream_t stream);


#endif // HEATMAP_CUDA_H
