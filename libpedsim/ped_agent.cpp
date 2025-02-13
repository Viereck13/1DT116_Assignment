//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
// Each Tagent object holds pointers to its data in SoA arrays.

#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>

// Static Arrays for SoA
namespace Ped {
    vector<int> Tagent::xs;
    vector<int> Tagent::ys;
    vector<int> Tagent::desiredXs;
    vector<int> Tagent::desiredYs;
    vector<Twaypoint*> Tagent::destinations;
    // vector<Twaypoint*> Tagent::lastDestinations;
    vector< deque<Twaypoint*> > Tagent::waypointQueues;
    vector<int> Tagent::destXs;
    vector<int> Tagent::destYs;

    size_t Tagent::nextIndex = 0;
}

namespace Ped {

    // New static method to reserve capacity for all agents.
    // No push_backs are allowed after this method is called because it will invalidate pointers.
    void Tagent::reserveAgents(size_t count) {
        xs.resize(count);
        ys.resize(count);
        desiredXs.resize(count);
        desiredYs.resize(count);
        destinations.resize(count);
        // lastDestinations.resize(count);
        waypointQueues.resize(count);

        destXs.resize(count);
        destYs.resize(count);
		nextIndex = 0;
		// std::cout << "reserveAgents: resized arrays to " << count << " elements." << std::endl;
    }

    Tagent::Tagent(int posX, int posY) {
        init(posX, posY);
    }

    Tagent::Tagent(double posX, double posY) {
        init((int)round(posX), (int)round(posY));
    }

    void Tagent::init(int posX, int posY) {
        // Use the pre-allocated slot at nextIndex.
        index = nextIndex++;
        
        xs[index] = posX;
        ys[index] = posY;
        desiredXs[index] = posX;  // initially, desired equals current position
        desiredYs[index] = posY;
        destinations[index] = NULL;
        // lastDestinations[index] = NULL;
        waypointQueues[index] = deque<Twaypoint*>();  // assign an empty deque

        destXs[index] = 0;
        destYs[index] = 0;
        
        // Store pointers to the data.
        pX = &xs[index];
        pY = &ys[index];
        pDesiredX = &desiredXs[index];
        pDesiredY = &desiredYs[index];
        pDestination = &destinations[index];

		// std::cout << "init: Agent at index " << index << " initialized with (" << xs[index] << ", " << ys[index] << ")." << std::endl;
		// std::cout << "init: Agent at index " << index << " initialized with (" << *pX << ", " << *pY << ")." << std::endl;
    }

    void Tagent::computeNextDesiredPosition() {
        Twaypoint* dest = getNextDestination();
        *pDestination = dest;
        if (dest == NULL) {
            // No destination available.
            return;
        }

        destXs[index] = dest->getx();
        destYs[index] = dest->gety();

        // cout << "Agent " << index << " at position " << *pX << ", " << *pY << " moving to " << dest->getx() << ", " << dest->gety() << endl;
        // cout << "Agent " << index << " at position " << *pX << ", " << *pY << " moving to " << destXs[index] << ", " << destYs[index] << endl;

        double diffX = dest->getx() - *pX;
        double diffY = dest->gety() - *pY;
        double len = sqrt(diffX * diffX + diffY * diffY);
        *pDesiredX = (int)round(*pX + diffX / len);
        *pDesiredY = (int)round(*pY + diffY / len);
    }

    void Tagent::computeNextDesiredPositionSIMD() {
        Twaypoint* dest = getNextDestination();
        *pDestination = dest;
        if (dest == NULL) {
            // No destination available.
            return;
        }

        destXs[index] = dest->getx();
        destYs[index] = dest->gety();

        // cout << "Agent " << index << " at position " << *pX << ", " << *pY << " moving to " << dest->getx() << ", " << dest->gety() << endl;
        // cout << "Agent " << index << " at position " << *pX << ", " << *pY << " moving to " << destXs[index] << ", " << destYs[index] << endl;

        // double diffX = dest->getx() - *pX;
        // double diffY = dest->gety() - *pY;
        // double len = sqrt(diffX * diffX + diffY * diffY);
        // *pDesiredX = (int)round(*pX + diffX / len);
        // *pDesiredY = (int)round(*pY + diffY / len);
    }

    void Tagent::addWaypoint(Twaypoint* wp) {
        waypointQueues[index].push_back(wp);
    }

    Twaypoint* Tagent::getNextDestination() {
        Twaypoint* currentDest = *pDestination;
        bool agentReachedDestination = false;
        if (currentDest != NULL) {
            double diffX = currentDest->getx() - *pX;
            double diffY = currentDest->gety() - *pY;
            double length = sqrt(diffX * diffX + diffY * diffY);
            agentReachedDestination = (length < currentDest->getr());
        }
        Twaypoint* nextDestination = NULL;
        if ((agentReachedDestination || currentDest == NULL) && !waypointQueues[index].empty()) {
            // Rotate the waypoint queue: push the current destination back...
            waypointQueues[index].push_back(currentDest);
            // ...and then pop the next one.
            nextDestination = waypointQueues[index].front();
            waypointQueues[index].pop_front();
        }
        else {
            nextDestination = currentDest;
        }
        return nextDestination;
    }
}