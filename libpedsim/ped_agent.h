//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. 
// The desired next position represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement is handled in ped_model.cpp.

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>
#include <cmath>
#include <stdlib.h>

using namespace std;

namespace Ped {

    class Twaypoint;

    class Tagent {
    public:
        Tagent(int posX, int posY);
        Tagent(double posX, double posY);

        // Returns the coordinates of the desired position
        int getDesiredX() const { return *pDesiredX; }
        int getDesiredY() const { return *pDesiredY; }

        // Sets the agent's position
        void setX(int newX) { *pX = newX; }
        void setY(int newY) { *pY = newY; }

        // Update the position (move the agents) to get closer to the current destination
        void computeNextDesiredPosition();
        void computeNextDesiredPositionSIMD();

        // Position of agent defined by x and y
		int getX() const { return *pX; }
        int getY() const { return *pY; }

        // Adds a new waypoint to reach this agent
        void addWaypoint(Twaypoint* wp);

		// Reserve capacity in the static arrays for a given number of agents.
		static void reserveAgents(size_t count);

        static int* getXsData() { return xs.data(); }
        static int* getYsData() { return ys.data(); }
        static int* getDesiredXsData() { return desiredXs.data(); }
        static int* getDesiredYsData() { return desiredYs.data(); }
        static int* getDestXsData() { return destXs.data(); }
        static int* getDestYsData() { return destYs.data(); }


    private:
        Tagent() {}  // disallow default construction

        // Each agent now only stores its coordinate into static arrays through its index.
        size_t index;
		// A static counter to track the next index for a new agent.
        static size_t nextIndex;

        int* pX;
        int* pY;
        int* pDesiredX;
        int* pDesiredY;
        Twaypoint** pDestination;

		// Static arrays (SoA layout) for all agent data.
        // If these vectors reallocate after initialization, the pointers become invalid! (1 day of debugging)
        static vector<int> xs;
        static vector<int> ys;
        static vector<int> desiredXs;
        static vector<int> desiredYs;
        static vector<Twaypoint*> destinations;
        // static vector<Twaypoint*> lastDestinations;
        static vector< deque<Twaypoint*> > waypointQueues;

        // ===== New for SIMD destination data =====
        static vector<int> destXs;   // holds the actual x coordinate of each agent's destination, not a pointer to
        static vector<int> destYs;   // holds the actual y coordinate of each agent's destination, not a pointer to

        // Internal init function 
        void init(int posX, int posY);

        // Returns the next destination to visit
        Twaypoint* getNextDestination();
    };

}

#endif