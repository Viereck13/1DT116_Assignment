#pragma once

struct CUDA_Waypoint
{
	double x;
	double y;
	double r;
};

struct CUDA_Agent
{
	int start;
	int size;
	int index;
};

class cuda_wrapper
{
	private:
public:
	CUDA_Waypoint* waypoint_device;
	CUDA_Agent* agent_device;

	int32_t* x_device;
	int32_t* y_device;
	
	double* destinationX_device;
	double* destinationY_device;
	double* destinationR_device;

	int numElements;

	void tick_CUDA();
	void setup_CUDA(
		std::vector<CUDA_Waypoint> &waypoints,
    	std::vector<CUDA_Agent> &agent_waypoints_indices,
		std::vector<int32_t> &x,
		std::vector<int32_t> &y
	);
	void writeBack_CUDA(int32_t *x, int32_t *y) const;
};


// void tick_CUDA(
// 	CUDA_Waypoint* waypoint_device,
// 	CUDA_Agent* agent_device,
// 	int32_t* x_device,
// 	int32_t* y_device,
// 	double* destinationX_device,
// 	double* destinationY_device,
// 	double* destinationR_device,
//     int numElements
// );

