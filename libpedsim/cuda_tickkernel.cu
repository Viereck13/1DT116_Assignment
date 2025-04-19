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