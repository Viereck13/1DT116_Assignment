#pragma once

void tick_CUDA(
	CUDA_Waypoint* waypoint_device,
	CUDA_Agent* agent_device,
	int32_t* x_device,
	int32_t* y_device,
	double* destinationX_device,
	double* destinationY_device,
	double* destinationR_device,
    int numElements
);