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
#include <cmath>

#include <emmintrin.h>
#include <immintrin.h>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#include "cuda_tickkernel.h"
#endif

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());
	// for (auto agent : agents) {
	// 	cout << "Agent: x: " << agent->getX() << ", y: " << agent->getY() << endl;
	// }

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;
	printf("Mode: %d\n", implementation);

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

	////// Collect all agents and distribute info to vectors
	if (implementation == CUDA || implementation == VECTOR) {
		for (auto agent : agents)
		{
			x.push_back(agent->getX());
			y.push_back(agent->getY());
			// desiredX.push_back(agent->getDesiredX());
			// desiredY.push_back(agent->getDesiredY());
			// printf("LOL\n");
			// if (agent->destination == NULL) {printf("R.I.P\n");};
			if (implementation == VECTOR) {
				Twaypoint* temp = agent->waypoints.front();
				destinationX.push_back(temp->getx());
				destinationY.push_back(temp->gety());
				destinationR.push_back(temp->getr());
				agent->destination = temp;
				agent->waypoints.pop_front();
			}
		}
		for (size_t i = 0; i < (4-(agents.size()%4))%4; i++)
		{
			x.push_back(0);
			y.push_back(0);
			// desiredX.push_back(0);
			// desiredY.push_back(0);
			destinationX.push_back(0.0);
			destinationY.push_back(0.0);
			destinationR.push_back(0.0);
		}

		#ifndef NOCDUA
		if (implementation == CUDA) {
			for (size_t i = 0; i < agents.size(); i++)
			{
				// printf("%d\t%d\n",(int)waypoints.size(),(int)agents.at(i)->waypoints.size());
				agent_waypoints_indices.push_back(CUDA_Agent{
					(int)waypoints.size(),
					(int)agents.at(i)->waypoints.size(),
					-1
				});
				for (size_t j = 0; j < agents.at(i)->waypoints.size(); j++)
				{
					waypoints.push_back(CUDA_Waypoint{
						agents.at(i)->waypoints.at(j)->getx(),
						agents.at(i)->waypoints.at(j)->gety(),
						agents.at(i)->waypoints.at(j)->getr()
					});
				}	
			}

			cuda.setup_CUDA(waypoints, agent_waypoints_indices, x, y);
			// printf("%d\t%d\n",(int)waypoints.size(),sizeof(CUDA_Waypoint));

			// cudaError_t err = cudaSuccess;

			
		}
		#endif
	}
	if(implementation == OMP){
		initRegions();
	}
	
}



// We assume that destionation is not NULL
void Ped::Model::tick_SIMD()
{
	#pragma omp parallel for
	for (size_t i = 0; i < agents.size(); i+=4)
	{
		// double diffX = destination->getx() - x;
		__m256d tick_x = _mm256_cvtepi32_pd(_mm_load_si128((__m128i*)&x[i]));
		__m256d tick_destination_x = _mm256_loadu_pd(&destinationX[i]);
		__m256d delta_x = _mm256_sub_pd(tick_destination_x, tick_x);
		
		// double diffY = destination->gety() - y;
		__m256d tick_y = _mm256_cvtepi32_pd(_mm_load_si128((__m128i*)&y[i]));
		__m256d tick_destination_y = _mm256_loadu_pd(&destinationY[i]);
		__m256d delta_y = _mm256_sub_pd(tick_destination_y, tick_y);

		// double length = sqrt(diffX * diffX + diffY * diffY);
		__m256d delta_x_sq = _mm256_mul_pd(delta_x, delta_x);
		__m256d delta_y_sq = _mm256_mul_pd(delta_y, delta_y);
		__m256d length = _mm256_sqrt_pd(_mm256_add_pd(delta_x_sq, delta_y_sq));
		
		// agentReachedDestination = length < destination->getr();
		__m256d tick_destination_r = _mm256_loadu_pd(&destinationR[i]);
		__m256i compare = _mm256_castpd_si256(_mm256_cmp_pd(length,tick_destination_r,_CMP_NGE_UQ));
		int agentReachedDestination = _mm256_testz_si256(compare,_mm256_set1_epi64x(1));

		// check if min one agent needs new destination
		if (agentReachedDestination == 0) {
			uint64_t agentsFlags[4];
			_mm256_storeu_si256((__m256i*)agentsFlags,compare);
			for (int j = 0; j < 4; j++) {
				if (agentsFlags[j] != 0 && i+j < agents.size()) {
					agents.at(i+j)->waypoints.push_back(agents.at(i+j)->destination);
					Ped::Twaypoint* ptw = agents.at(i+j)->waypoints.front();
					agents.at(i+j)->destination = ptw; // update agent
					// update your local data vector
					destinationX.at(i+j) = ptw->getx();
					destinationY.at(i+j) = ptw->gety();
					destinationR.at(i+j) = ptw->getr();
					agents.at(i+j)->waypoints.pop_front();
				}
			}

			// If new destionation have been choosen -> recalculate!
			// double diffX = destination->getx() - x;
			tick_destination_x = _mm256_loadu_pd(&destinationX[i]);
			delta_x = _mm256_sub_pd(tick_destination_x, tick_x);
			
			// double diffY = destination->gety() - y;
			tick_destination_y = _mm256_loadu_pd(&destinationY[i]);
			delta_y = _mm256_sub_pd(tick_destination_y, tick_y);

			// double length = sqrt(diffX * diffX + diffY * diffY);
			delta_x_sq = _mm256_mul_pd(delta_x, delta_x);
			delta_y_sq = _mm256_mul_pd(delta_y, delta_y);
			length = _mm256_sqrt_pd(_mm256_add_pd(delta_x_sq, delta_y_sq));
		}

		// double diffX = destination->getx() - x;
		// double diffY = destination->gety() - y;
		// double len = sqrt(diffX * diffX + diffY * diffY);

		// desiredPositionX = (int)round(x + diffX / len);
		__m256d tick_depox_s1 = _mm256_div_pd(delta_x, length);
		__m256d tick_depox_s2 = _mm256_add_pd(tick_x, tick_depox_s1);
		__m128i tick_depox_s3 = _mm256_cvtpd_epi32(_mm256_round_pd(tick_depox_s2,_MM_FROUND_TO_NEAREST_INT));
		_mm_store_si128((__m128i*)&x[i], tick_depox_s3);
		// desiredPositionY = (int)round(y + diffY / len);
		__m256d tick_depoy_s1 = _mm256_div_pd(delta_y, length);
		__m256d tick_depoy_s2 = _mm256_add_pd(tick_y, tick_depoy_s1);
		__m128i tick_depoy_s3 = _mm256_cvtpd_epi32(_mm256_round_pd(tick_depoy_s2,_MM_FROUND_TO_NEAREST_INT));
		_mm_store_si128((__m128i*)&y[i], tick_depoy_s3);
	}
}

// void Ped::Model::tick_OMP() {
// 	#pragma omp parallel for schedule(static) 
// 	for (int i = 0; i < agents.size(); i++) {
// 		Ped::Tagent* agent = agents[i]; 
// 		agent->computeNextDesiredPosition();
// 		agent->setX(agent->getDesiredX());
// 		agent->setY(agent->getDesiredY());
// 	}
// }

void Ped::Model::tick_OMP()
{

    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<agents.size(); i++)
    {
		// cout << "Agent: " << i << " " << agents[i]->getX() << " " << agents[i]->getY() << endl;
        agents[i]->computeNextDesiredPosition();
		// cout << "Desired: " << agents[i]->getDesiredX() << " " << agents[i]->getDesiredY() << endl;
    }

    #pragma omp parallel for schedule(static, 1) num_threads(regions.size())
    for (int r=0; r<regions.size(); r++)
    {
		// if(omp_get_thread_num() == 0){
		// 	printf("Number of threads: %d\n", omp_get_num_threads());
		// }
        omp_set_lock(&regions[r]->lock);
        std::vector<Tagent*> localAgents = regions[r]->agents;
        omp_unset_lock(&regions[r]->lock);

        // Process each agent in the local copy.
        for (auto agent : localAgents)
        {
			move_OMP(agent);
        }
    }
	
}

void Ped::Model::tick_CTHREADS() {
	std::vector<std::thread> threads;

    // Lambda function to update a subset of agents
    auto updateAgents = [](std::vector<Ped::Tagent*>::iterator start, std::vector<Ped::Tagent*>::iterator end) {
        for (auto it = start; it != end; it++) {
            Ped::Tagent* agent = *it;

            agent->computeNextDesiredPosition();
            agent->setX(agent->getDesiredX());
            agent->setY(agent->getDesiredY());
        }
    };

    // Determine the number of agents per thread, and the number of threads will be limited by the hardware.
    const unsigned int numThreads = std::thread::hardware_concurrency(); // Returns the number of concurrent threads supported by the implementation.
    const unsigned int agentsPerThread = agents.size() / numThreads;
	cout << "Number of threads: " << numThreads << endl;
	cout << "Number of agents: " << agents.size() << endl;
	cout << "Agents per thread: " << agentsPerThread << endl;
    

	auto agentStart = agents.begin();
    for (int i = 0; i < numThreads; i++) {
        auto agentEnd = (i == numThreads-1) ? agents.end() : agentStart + agentsPerThread;
        threads.emplace_back(updateAgents, agentStart, agentEnd);
        agentStart = agentEnd; // The first agent of the next thread is the last agent of the current thread
    }

    // Join threads
    for (std::thread& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void Ped::Model::tick()
{
	switch (implementation)
	{
	case OMP:
		tick_OMP();
		break;
	case PTHREAD:
		tick_CTHREADS();
		break;
	case VECTOR:
		tick_SIMD();
		break;
	case CUDA:
		cuda.tick_CUDA();
		break;
	case SEQ:
		for (auto agent : agents)
		{
			agent->computeNextDesiredPosition();
			// agent->setX(agent->getDesiredX());
			// agent->setY(agent->getDesiredY());
			move(agent);
		}
		break;
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// New helper: initialize regions
void Ped::Model::initRegions()
{
    int regionWidth = worldWidth / regionsX;
    int regionHeight = worldHeight / regionsY;
    int idCounter = 0;
    for (int i = 0; i < regionsY; i++) {
        for (int j = 0; j < regionsX; j++) {
            int minX = j * regionWidth;
            int maxX = (j+1) * regionWidth;
            int minY = i * regionHeight;
            int maxY = (i+1) * regionHeight;
			// cout << "Region " << idCounter << ": " << minX << " " << maxX << " " << minY << " " << maxY << endl;
            regions.push_back(new Region(idCounter++, minX, maxX, minY, maxY));
        }
    }
	// cout << "Regions: " << regions.size() << endl;
    // After setting up regions, distribute agents into regions
    for (auto agent : agents) {
		// cout << "Agent " << agent->getX() << " " << agent->getY() << endl;
        Region* r = getRegionForPosition(agent->getX(), agent->getY());
		// std::cout << "Agent " << agent->getX() << " " << agent->getY() << " in region " << r->id << std::endl;
        r->agents.push_back(agent);
    }
}

// Get a region for a given position
Ped::Region* Ped::Model::getRegionForPosition(int x, int y)
{
	if (x<0) x = 0;
	else if (x>=worldWidth) x = worldWidth-1;
	if (y<0) y = 0;
	else if (y>=worldHeight) y = worldHeight-1;
    for (auto r : regions) {
        if (x >= r->minX && x < r->maxX &&
            y >= r->minY && y < r->maxY)
        {
            return r;
        }
    }
	// cout << "--------getRegionForPosition--------" << endl;
	// cout << x << " " << y << endl;
	// cout << "Region not found" << endl;
    return nullptr; // should not happen if regions cover the entire world.
}

bool Ped::Model::withinRegion(int x, int y)
{
	if (x<0) x = 0;
	else if (x>=worldWidth) x = worldWidth-1;
	if (y<0) y = 0;
	else if (y>=worldHeight) y = worldHeight-1;
    for (auto r : regions) {
        if (x >= r->minX+1 && x < r->maxX-1 &&
            y >= r->minY+1 && y < r->maxY-1)
        {
            return true;
        }
    }
	// cout << "--------getRegionForPosition--------" << endl;
	// cout << x << " " << y << endl;
	// cout << "Region not found" << endl;
    return false; // should not happen if regions cover the entire world.
}

void Ped::Model::move_OMP(Ped::Tagent *agent)
{
    int curX = agent->getX();
    int curY = agent->getY();
    int desiredX = agent->getDesiredX();
    int desiredY = agent->getDesiredY();

    // Get positions of neighboring agents
    std::set<const Ped::Tagent*> neighbors = getNeighbors(curX, curY, 2);
    std::vector<std::pair<int, int>> takenPositions;
    for (auto neighbor : neighbors) {
        takenPositions.push_back(std::make_pair(neighbor->getX(), neighbor->getY()));
    }

    // Create a prioritized list of candidate moves.
    std::vector<std::pair<int, int>> prioritizedAlternatives;
    std::pair<int, int> pDesired(desiredX, desiredY);
    prioritizedAlternatives.push_back(pDesired);

    int diffX = desiredX - curX;
    int diffY = desiredY - curY;
    std::pair<int, int> p1, p2;
    if (diffX == 0 || diffY == 0) {
        // Agent walks straight: N, S, E, or W
        p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
        p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
    }
    else {
        // Agent walks diagonally: try horizontal and vertical alternatives.
        p1 = std::make_pair(pDesired.first, curY);
        p2 = std::make_pair(curX, pDesired.second);
    }
    prioritizedAlternatives.push_back(p1);
    prioritizedAlternatives.push_back(p2);

    // Choose a move from the prioritized list.
    std::pair<int, int> chosenPos = std::make_pair(curX, curY);
    bool candidateFound = false;
    for (auto &pos : prioritizedAlternatives) {
        if (std::find(takenPositions.begin(), takenPositions.end(), pos) == takenPositions.end()) {
            chosenPos = pos;
            candidateFound = true;
            break;
        }
    }
    if (!candidateFound) {
        // Check all adjacent cells.
        std::vector<std::pair<int, int>> adjacent;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0)
                    continue;
                adjacent.push_back(std::make_pair(curX + dx, curY + dy));
            }
        }
        std::vector<std::pair<int, int>> freeAdjacent;
        for (auto &pos : adjacent) {
            if (std::find(takenPositions.begin(), takenPositions.end(), pos) == takenPositions.end())
                freeAdjacent.push_back(pos);
        }
        if (!freeAdjacent.empty()) {
            auto best = freeAdjacent.begin();
            double bestDist = sqrt(pow(best->first - desiredX, 2) + pow(best->second - desiredY, 2));
            for (auto it = freeAdjacent.begin(); it != freeAdjacent.end(); ++it) {
                double dist = sqrt(pow(it->first - desiredX, 2) + pow(it->second - desiredY, 2));
                if (dist < bestDist) {
                    bestDist = dist;
                    best = it;
                }
            }
            chosenPos = *best;
            candidateFound = true;
        }
        // If no free adjacent cell is found, the agent remains in place.
    }

    // Determine region membership before and after the move.
    Region* curRegion = getRegionForPosition(agent->getX(), agent->getY());
    Region* targetRegion = getRegionForPosition(chosenPos.first, chosenPos.second);

    if (withinRegion(chosenPos.first, chosenPos.second)) {
        // Even for an intraregion move, we lock the region
        // omp_set_lock(&curRegion->lock);
        // {
            // Re-check that the candidate cell is still free
            bool conflict = false;
            for (auto otherAgent : curRegion->agents) {
                if (otherAgent != agent && otherAgent->getX() == chosenPos.first && otherAgent->getY() == chosenPos.second) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                // Update the agent's position.
                agent->setX(chosenPos.first);
                agent->setY(chosenPos.second);
            }
            // If there’s a conflict, leave the agent in place
        // }
        // omp_unset_lock(&curRegion->lock);
    }
    else if (curRegion == targetRegion) {
        // Even for an intraregion move, we lock the region
        omp_set_lock(&curRegion->lock);
        {
            // Recheck that the candidate cell is still free
            bool conflict = false;
            for (auto otherAgent : curRegion->agents) {
                if (otherAgent != agent && otherAgent->getX() == chosenPos.first && otherAgent->getY() == chosenPos.second) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                // Update the agent's position.
                agent->setX(chosenPos.first);
                agent->setY(chosenPos.second);
            }
            // If there’s a conflict, leave the agent in place
        }
        omp_unset_lock(&curRegion->lock);
    }
    else {
        // Border move: lock both regions in consistent order
        if (curRegion->id < targetRegion->id) {
            omp_set_lock(&curRegion->lock);
            omp_set_lock(&targetRegion->lock);
        } else {
            omp_set_lock(&targetRegion->lock);
            omp_set_lock(&curRegion->lock);
        }

        // Re-check that chosenPos is still free in both regions.
        bool conflict = false;
        for (auto otherAgent : curRegion->agents) {
            if (otherAgent != agent && otherAgent->getX() == chosenPos.first && otherAgent->getY() == chosenPos.second) {
                conflict = true;
                break;
            }
        }
        if (!conflict) {
            for (auto otherAgent : targetRegion->agents) {
                if (otherAgent->getX() == chosenPos.first && otherAgent->getY() == chosenPos.second) {
                    conflict = true;
                    break;
                }
            }
        }

        if (!conflict) {
            // Remove agent from the source region’s list.
            auto &srcAgents = curRegion->agents;
            srcAgents.erase(std::remove(srcAgents.begin(), srcAgents.end(), agent), srcAgents.end());

            // Update agent’s coordinates.
            agent->setX(chosenPos.first);
            agent->setY(chosenPos.second);

            // Add agent to the target region’s list.
            targetRegion->agents.push_back(agent);
        }
        // Otherwise, leave the agent in place

        // Unlock in reverse order.
        if (curRegion->id < targetRegion->id) {
            omp_unset_lock(&targetRegion->lock);
            omp_unset_lock(&curRegion->lock);
        } else {
            omp_unset_lock(&curRegion->lock);
            omp_unset_lock(&targetRegion->lock);
        }
    }
}


void Ped::Model::move(Ped::Tagent *agent)
{

    int curX = agent->getX();
    int curY = agent->getY();
    int desiredX = agent->getDesiredX();
    int desiredY = agent->getDesiredY();

	// Gather positions of neighboring agents.
	std::set<const Ped::Tagent*> neighbors = getNeighbors(curX, curY, 2);
	std::vector<std::pair<int, int>> takenPositions;
	for (auto neighbor : neighbors)
	{
		takenPositions.push_back(std::make_pair(neighbor->getX(), neighbor->getY()));
	}

    // Create a prioritized list of candidate moves.
    std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
    prioritizedAlternatives.push_back(std::make_pair(desiredX, desiredY));

    int diffX = desiredX-curX;
    int diffY = desiredY-curY;
    	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent walks straight to N, S, W or E
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent walks diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

    // Check the prioritized candidate positions.
    for (auto &pos : prioritizedAlternatives)
    {
        if (std::find(takenPositions.begin(), takenPositions.end(), pos) == takenPositions.end())
        {
            agent->setX(pos.first);
            agent->setY(pos.second);
            return;
        }
    }
	// If the code reaches here, it means that all three prioritized moves are not possible.
    // Examine all 8 adjacent cells. Similar to BFS, find the closest free cell to move to. Not limited by the 3 prioritized moves anymore.
    std::vector<std::pair<int, int>> adjacent;
    for (int dx=-1; dx<=1; dx++)
    {
        for (int dy=-1; dy<=1; dy++)
        {
            // Skip the current cell. You want to move!
            if (dx == 0 && dy == 0)
                continue;
            adjacent.push_back(std::make_pair(curX+dx, curY+dy));
        }
    }

    // Filter out only the free adjacent cells.
    std::vector<std::pair<int, int>> freeAdjacent;
    for (auto &pos : adjacent)
    {
        if (std::find(takenPositions.begin(), takenPositions.end(), pos) == takenPositions.end())
        {
            freeAdjacent.push_back(pos);
        }
    }

    if (!freeAdjacent.empty())
    {
        // Choose the free adjacent cell that minimizes the distance to the desired position.
        auto best = freeAdjacent.begin();
        double bestDist = sqrt(pow(best->first - desiredX, 2) + pow(best->second - desiredY, 2));
        for (auto it = freeAdjacent.begin(); it != freeAdjacent.end(); it++)
        {
            double dist = sqrt(pow(it->first - desiredX, 2) + pow(it->second - desiredY, 2));
            if (dist<bestDist)
            {
                bestDist = dist;
                best = it;
            }
        }
        agent->setX(best->first);
        agent->setY(best->second);
        return;
    }

    // If no adjacent cell is free, the agent remains in its current position.
	// Wishing that the deadlock will be resolved in the next time step, which some agents around this agent will move.
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
// set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

// 	// create the output list
// 	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
// 	return set<const Ped::Tagent*>(agents.begin(), agents.end());
// }

// 
// Only returns agents within the specified distance (square area)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
    set<const Ped::Tagent*> result;
    for (auto agent : agents) {
         int dx = agent->getX() - x;
         int dy = agent->getY() - y;
         if (abs(dx) <= dist && abs(dy) <= dist) {
              result.insert(agent);
         }
    }
    return result;
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
