#include "ped_model.h"

__global__ void agentTickKernel(int32_t *x, int32_t *y, double *destinationX, double *destinationY, double *destinationR, CUDA_Waypoint *waypoints, CUDA_Agent *agents, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElements) return;

    // Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	// if (destination != NULL) {
    if (agents[i].index >= 0) {

		// compute if agent reached its current destination
        // 	double diffX = destination->getx() - x;
        double diffX = destinationX[i] - x[i];
        // 	double diffY = destination->gety() - y;
        double diffY = destinationY[i] - y[i];
        // 	double length = sqrt(diffX * diffX + diffY * diffY);
        double length = sqrt(diffX * diffX + diffY * diffY);
        // 	agentReachedDestination = length < destination->getr();
        agentReachedDestination = length < destinationR[i];
        // }
    }

	if ((agentReachedDestination || agents[i].index < 0) && agents[i].size != 0) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		// waypoints.push_back(destination);
		// nextDestination = waypoints.front();
		// waypoints.pop_front();
        agents[i].index += 1;
        if (agents[i].index >= agents[i].size) {
            agents[i].index = -1;
        } else if (agents[i].index >= 0) {
            destinationX[i] = waypoints[agents[i].start+agents[i].index].x;
            destinationY[i] = waypoints[agents[i].start+agents[i].index].y;
            destinationR[i] = waypoints[agents[i].start+agents[i].index].r;
        }
	}

	if (agents[i].index >= 0) {
		// no destination, no need to
		// compute where to move to
		// return;
        
        double diffX = destinationX[i] - x[i];
        double diffY = destinationY[i] - y[i];
        double len = sqrt(diffX * diffX + diffY * diffY);
        x[i] = (int)round(x[i] + diffX / len);
        y[i] = (int)round(y[i] + diffY / len);
	}
}

void cuda_wrapper::tick_CUDA() {
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
    //      threadsPerBlock);
    agentTickKernel<<<blocksPerGrid, threadsPerBlock>>>(
        x_device,
        y_device,
        destinationX_device,
        destinationY_device,
        destinationR_device,
        waypoint_device,
        agent_device,
        numElements);
}

void cuda_wrapper::setup_CUDA(
    std::vector<CUDA_Waypoint> &waypoints,
    std::vector<CUDA_Agent> &agent_waypoints_indices,
    std::vector<int32_t> &x,
    std::vector<int32_t> &y
) {
    numElements = agent_waypoints_indices.size();
    cudaMalloc(&waypoint_device, waypoints.size()*sizeof(CUDA_Waypoint));
    cudaMemcpy(waypoint_device, waypoints.data(),  waypoints.size()*sizeof(CUDA_Waypoint), cudaMemcpyHostToDevice);
    cudaMalloc(&agent_device, numElements*sizeof(CUDA_Agent));
    cudaMemcpy(agent_device, agent_waypoints_indices.data(),  numElements*sizeof(CUDA_Agent), cudaMemcpyHostToDevice);

    cudaMalloc(&x_device, numElements*sizeof(int32_t));
    cudaMemcpy(x_device, x.data(),  numElements*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMalloc(&y_device, numElements*sizeof(int32_t));
    cudaMemcpy(y_device, y.data(),  numElements*sizeof(int32_t), cudaMemcpyHostToDevice);
    
    cudaMalloc(&destinationX_device, numElements*sizeof(double));
    // cudaMemcpy(destinationX_device, destinationX.data(),  destinationX.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&destinationY_device, numElements*sizeof(double));
    // cudaMemcpy(destinationY_device, destinationY.data(),  destinationY.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&destinationR_device, numElements*sizeof(double));
    // cudaMemcpy(destinationR_device, destinationR.data(),  destinationR.size()*sizeof(double), cudaMemcpyHostToDevice);
}

void cuda_wrapper::writeBack_CUDA(int32_t *x, int32_t *y) const {
    cudaMemcpy((void*)x, x_device, numElements*sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)y, y_device, numElements*sizeof(int32_t), cudaMemcpyDeviceToHost);
}

/******/
__global__ void heatmapTickKernel(int* heatmap_d, int* scaled_heatmap_d, int* blurred_heatmap_d, int numAgents, int* desiredX, int* desiredY);

void cuda_wrapper::tickHeatmap() {
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (SCALED_SIZE*SCALED_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
    //      threadsPerBlock);
    heatmapTickKernel<<<blocksPerGrid, threadsPerBlock>>>(
        heatmap_device, scaled_heatmap_device, blurred_heatmap_device, numElements, desiredX_device, desiredY_device);
}

void cuda_wrapper::setupHeatmap()
{
    cudaMalloc(&heatmap_device, SIZE*SIZE*sizeof(int));
    cudaMemset(&heatmap_device, 0, SIZE*SIZE*sizeof(int));
    cudaMalloc(&scaled_heatmap_device, SCALED_SIZE*SCALED_SIZE*sizeof(int));
    cudaMalloc(&blurred_heatmap_device, SCALED_SIZE*SCALED_SIZE*sizeof(int));
}

void cuda_wrapper::writeBackHeatmap(int** heatmap, int** scaled_heatmap, int** blurred_heatmap)
{
    int* heatmap_l;
    int* scaled_heatmap_l;
    int* blurred_heatmap_l;

    cudaMallocHost(&heatmap_l, SIZE*SIZE*sizeof(int));
    cudaMemset(&heatmap_l, 0, SIZE*SIZE*sizeof(int));
    cudaMallocHost(&scaled_heatmap_l, SCALED_SIZE*SCALED_SIZE*sizeof(int));
    cudaMallocHost(&blurred_heatmap_l, SCALED_SIZE*SCALED_SIZE*sizeof(int));

    cudaMemcpy(&heatmap_l, &heatmap_device,  SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&scaled_heatmap_l, &scaled_heatmap_device, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&blurred_heatmap_l, &blurred_heatmap_device, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < SIZE*SIZE; i++)
    {
        heatmap[i/SIZE][i%SIZE] = heatmap_l[i];
    }
    
    // std::cout << "PROBLEM: " << sizeof(scaled_heatmap) << std::endl;
    for (size_t i = 0; i < SCALED_SIZE*SCALED_SIZE; i++)
    {
        scaled_heatmap[i/SCALED_SIZE][i%SCALED_SIZE] = scaled_heatmap_l[i];
        blurred_heatmap[i/SCALED_SIZE][i%SCALED_SIZE] = blurred_heatmap_l[i];
    }
}

__global__ void heatmapTickKernel(int* heatmap_d, int* scaled_heatmap_d, int* blurred_heatmap_d, int numAgents, int* desiredX, int* desiredY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= numElements) return;
    if (i < SIZE*SIZE) {
        heatmap_d[i] = (int)round(heatmap_d[i] * 0.80);
    }

    __syncthreads();

    if (i < numAgents)
    {
        int x = desiredX[i];
        int y = desiredY[i];
    
        if (x >= 0 || x < SIZE || y >= 0 || y < SIZE)
        {
            atomicAdd(&heatmap_d[y*SIZE+x], 40);
        }
    }
    
    __syncthreads();

    if (i < SIZE*SIZE)
	{
        heatmap_d[i] = heatmap_d[i] < 255 ? heatmap_d[i] : 255;
	}
    
    __syncthreads();

    // Scale the data for visual representation
    // __shared__ int scaled_heatmap_d_s[SCALED_SIZE*SCALED_SIZE];
    if (i < SIZE*SIZE)
    {
        int value = heatmap_d[i];
        for (size_t j = 0; j < CELLSIZE; j++)
        {
            scaled_heatmap_d[i*CELLSIZE+j] = value;
        }
    }

    __syncthreads();

    // Weights for blur filter
    const int w[5][5] = {
        { 1, 4, 7, 4, 1 },
        { 4, 16, 26, 16, 4 },
        { 7, 26, 41, 26, 7 },
        { 4, 16, 26, 16, 4 },
        { 1, 4, 7, 4, 1 }
    };

    #define WEIGHTSUM 273
    
    if (i < SCALED_SIZE*SCALED_SIZE) {
        int x = i/SCALED_SIZE;
        int y = i%SCALED_SIZE;
        if (x < 2 || x >= SCALED_SIZE-2) {
            return;
        }
        if (y < 2 || y >= SCALED_SIZE-2) {
            return;
        }

        int sum = 0;
        for (int k = -2; k < 3; k++)
        {
            for (int l = -2; l < 3; l++)
            {
                sum += w[2 + k][2 + l] * scaled_heatmap_d[(x + k)*SCALED_SIZE + y + l];
            }
        }
        int value = sum / WEIGHTSUM;
        blurred_heatmap_d[x * SCALED_SIZE + y] = 0x00FF0000 | value << 24;
    }
}