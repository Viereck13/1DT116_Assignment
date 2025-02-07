#include <vector>
#include "ped_agent.h"

class agents
{
private:
    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> desiredX;
    std::vector<int> desiredY;

    std::vector<double> destinationX;
    std::vector<double> destinationX;

    std::vector<std::vector<Ped::Twaypoint>> waypoints;
    std::vector<size_t> waypointsIndex;

public:
    agents(/* args */);
    ~agents();
};

agents::agents(/* args */)
{
}
