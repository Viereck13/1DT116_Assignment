//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
// #include <xmmintrin.h>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

int cnt = 1;

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif

	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::tick()
{
	// std::cout << "The length of the vector is: " << agents.size() << std::endl;
	// Size is 453
	// EDIT HERE FOR ASSIGNMENT 1
    if(implementation == Ped::SEQ) {
        // printf("Running SEQ version: %d\n", cnt++);
        for (auto agent : agents) {
            // Let the agent calculate the desired position based on its destination.
            agent->computeNextDesiredPosition();

            // Retrieve the desired position.
            int newX = agent->getDesiredX();
            int newY = agent->getDesiredY();

            // Directly update the agent's current position.
            agent->setX(newX);
            agent->setY(newY);
        }
    } else if(implementation == Ped::OMP) {
        // printf("Running OMP version: %d\n", cnt++);
        #pragma omp parallel for schedule(static,5)
        for (int i = 0; i < agents.size(); i++) {
            Ped::Tagent* agent = agents[i]; 
            agent->computeNextDesiredPosition();
            // printf("Thread %d is processing agent %d\n", omp_get_thread_num(), i);
            
            agent->setX(agent->getDesiredX());
            agent->setY(agent->getDesiredY());
        }

    } else if(implementation == Ped::VECTOR){
        size_t numAgents = agents.size();

        // Update each agent’s destination and compute its desired position.
        // This ensures that destXs and destYs are up to date.
        for (size_t i = 0; i < numAgents; i++) {
            agents[i]->computeNextDesiredPositionSIMD();
        }

        size_t i = 0;
        // A small epsilon to avoid division by zero.
        __m128 epsilon = _mm_set1_ps(1e-6f);
        __m128 zero = _mm_setzero_ps();

        // Process agents in blocks of 4 using SSE
        for (; i+3 < numAgents; i+=4) {
            // Load current positions and convert to float.
            __m128i curXi = _mm_load_si128((__m128i*)(Tagent::getXsData() + i));
            __m128 curX = _mm_cvtepi32_ps(curXi);
            __m128i curYi = _mm_load_si128((__m128i*)(Tagent::getYsData() + i));
            __m128 curY = _mm_cvtepi32_ps(curYi);

            // Load destination positions and convert to float.
            __m128i destXi = _mm_load_si128((__m128i*)(Tagent::getDestXsData() + i));
            __m128 destX = _mm_cvtepi32_ps(destXi);
            __m128i destYi = _mm_load_si128((__m128i*)(Tagent::getDestYsData() + i));
            __m128 destY = _mm_cvtepi32_ps(destYi);

            // Compute differences: diff = destination - current.
            __m128 diffX = _mm_sub_ps(destX, curX);
            __m128 diffY = _mm_sub_ps(destY, curY);

            // Compute the squares of the differences.
            __m128 diffX2 = _mm_mul_ps(diffX, diffX);
            __m128 diffY2 = _mm_mul_ps(diffY, diffY);

            // Compute the sum of squares.
            __m128 sumSq = _mm_add_ps(diffX2, diffY2);

            // Compute the length.
            __m128 len = _mm_sqrt_ps(sumSq);

            // Create a mask: mask = (len > epsilon)
            __m128 mask = _mm_cmpgt_ps(len, epsilon);

            // For lanes where len is too small, substitute epsilon to avoid division by zero.
            __m128 safeLen = _mm_blendv_ps(epsilon, len, mask);

            // Compute normalized differences (diff / safeLen).
            __m128 normDiffX = _mm_div_ps(diffX, safeLen);
            __m128 normDiffY = _mm_div_ps(diffY, safeLen);

            // Now, if the length was below epsilon, normalize step to be zero.
            normDiffX = _mm_blendv_ps(zero, normDiffX, mask);
            normDiffY = _mm_blendv_ps(zero, normDiffY, mask);

            // Compute new desired positions: current + normalized difference.
            __m128 newX = _mm_add_ps(curX, normDiffX);
            __m128 newY = _mm_add_ps(curY, normDiffY);

            // Convert the computed positions back to integers.
            __m128i desiredXi = _mm_cvtps_epi32(newX);
            __m128i desiredYi = _mm_cvtps_epi32(newY);

            // Store the computed desired positions.
            _mm_store_si128((__m128i*)(Tagent::getDesiredXsData() + i), desiredXi);
            _mm_store_si128((__m128i*)(Tagent::getDesiredYsData() + i), desiredYi);
        }

        // Process any remaining agents that are the remaining of a block of 4.
        for (; i < numAgents; i++) {
            int cur = Tagent::getXsData()[i];
            int curYVal = Tagent::getYsData()[i];
            int dX = Tagent::getDestXsData()[i];
            int dY = Tagent::getDestYsData()[i];
            float diffX = dX - cur;
            float diffY = dY - curYVal;
            float length = sqrtf(diffX * diffX + diffY * diffY);
            if (length < 1e-6f) {
                Tagent::getDesiredXsData()[i] = cur;
                Tagent::getDesiredYsData()[i] = curYVal;
            } else {
                int desiredX = (int)roundf(cur + diffX / length);
                int desiredY = (int)roundf(curYVal + diffY / length);
                Tagent::getDesiredXsData()[i] = desiredX;
                Tagent::getDesiredYsData()[i] = desiredY;
            }
        }

        // Update each agent's current position with its computed desired position.
        for (size_t j = 0; j < numAgents; j++) {
            Tagent::getXsData()[j] = Tagent::getDesiredXsData()[j];
            Tagent::getYsData()[j] = Tagent::getDesiredYsData()[j];
        }
    } else if(implementation == Ped::OMP_SIMD) {
        size_t numAgents = agents.size();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numAgents; i++) {
            agents[i]->computeNextDesiredPositionSIMD();
        }

        // Determine the number of agents that can be processed in blocks of 4.
        size_t simdLoopEnd = (numAgents / 4) * 4;

        __m128 epsilon = _mm_set1_ps(1e-6f);
        __m128 zero    = _mm_setzero_ps();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < simdLoopEnd; i += 4) {
            __m128i curXi = _mm_load_si128((__m128i*)(Tagent::getXsData() + i));
            __m128  curX  = _mm_cvtepi32_ps(curXi);

            __m128i curYi = _mm_load_si128((__m128i*)(Tagent::getYsData() + i));
            __m128  curY  = _mm_cvtepi32_ps(curYi);

            __m128i destXi = _mm_load_si128((__m128i*)(Tagent::getDestXsData() + i));
            __m128  destX  = _mm_cvtepi32_ps(destXi);

            __m128i destYi = _mm_load_si128((__m128i*)(Tagent::getDestYsData() + i));
            __m128  destY  = _mm_cvtepi32_ps(destYi);

            __m128 diffX = _mm_sub_ps(destX, curX);
            __m128 diffY = _mm_sub_ps(destY, curY);

            __m128 diffX2 = _mm_mul_ps(diffX, diffX);
            __m128 diffY2 = _mm_mul_ps(diffY, diffY);

            __m128 sumSq = _mm_add_ps(diffX2, diffY2);
            __m128 len   = _mm_sqrt_ps(sumSq);

            __m128 mask = _mm_cmpgt_ps(len, epsilon);

            __m128 safeLen = _mm_blendv_ps(epsilon, len, mask);

            __m128 normDiffX = _mm_div_ps(diffX, safeLen);
            __m128 normDiffY = _mm_div_ps(diffY, safeLen);

            normDiffX = _mm_blendv_ps(zero, normDiffX, mask);
            normDiffY = _mm_blendv_ps(zero, normDiffY, mask);

            __m128 newX = _mm_add_ps(curX, normDiffX);
            __m128 newY = _mm_add_ps(curY, normDiffY);

            __m128i desiredXi = _mm_cvtps_epi32(newX);
            __m128i desiredYi = _mm_cvtps_epi32(newY);

            _mm_store_si128((__m128i*)(Tagent::getDesiredXsData() + i), desiredXi);
            _mm_store_si128((__m128i*)(Tagent::getDesiredYsData() + i), desiredYi);
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = simdLoopEnd; i < numAgents; i++) {
            int cur   = Tagent::getXsData()[i];
            int curYVal = Tagent::getYsData()[i];
            int dX    = Tagent::getDestXsData()[i];
            int dY    = Tagent::getDestYsData()[i];
            float diffX = dX - cur;
            float diffY = dY - curYVal;
            float length = sqrtf(diffX * diffX + diffY * diffY);
            if (length < 1e-6f) {
                Tagent::getDesiredXsData()[i] = cur;
                Tagent::getDesiredYsData()[i] = curYVal;
            } else {
                int desiredX = (int)roundf(cur + diffX / length);
                int desiredY = (int)roundf(curYVal + diffY / length);
                Tagent::getDesiredXsData()[i] = desiredX;
                Tagent::getDesiredYsData()[i] = desiredY;
            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < numAgents; j++) {
            Tagent::getXsData()[j] = Tagent::getDesiredXsData()[j];
            Tagent::getYsData()[j] = Tagent::getDesiredYsData()[j];
        }
    }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {
			agent->setX((*it).first);
			agent->setY((*it).second);
			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	// std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){cout << "Coordinates: " << agent->getX() << ", " << agent->getY() << endl;});
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}