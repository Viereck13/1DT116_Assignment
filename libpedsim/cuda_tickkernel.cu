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

void tick_CUDA(
	CUDA_Waypoint* waypoint_device,
	CUDA_Agent* agent_device,
	int32_t* x_device,
	int32_t* y_device,
	double* destinationX_device,
	double* destinationY_device,
	double* destinationR_device,
    int numElements
) {
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