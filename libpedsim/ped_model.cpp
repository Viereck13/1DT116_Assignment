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

#ifndef NOCDUA
#include "cuda_testkernel.h"
#include "cuda_tickkernel.h"
#endif

#include <stdlib.h>
#include <atomic>
#include <array>

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

	for (size_t i = 0; i < 8; i++)
	{
		Region regio;
		regio.leftBorder = i*(REGION_X_SIZE/ 8);
		for (size_t j = 0; j < REGION_Y_SIZE; j++)
			{
				//std::atomic<int>* atom = new std::atomic(-1);
				regio.leftBorderGates[j]->store(-1);
			}
		regio.rightBorder = (i+1)*(REGION_X_SIZE/ 8);
		for (size_t j = 0; j < REGION_Y_SIZE; j++)
			{
				//std::atomic<int>* atom = new std::atomic(-1);
				regio.rightBorderGates[j]->store(-1);
			}
		regio.assignedAgents;// = new std::vector<Ped::Tagent*>();
		// regio.changeRegion = new std::mutex();
		regions.push_back(regio);
	}
	for (size_t aT = 0; aT < agents.size(); aT++)
	{
		for (size_t i = 0; i < regions.size(); i++)
		{
			if (agents.at(aT)->getX() >= regions.at(i).leftBorder && agents.at(aT)->getX() < regions.at(i).rightBorder) {
				regions.at(i).assignedAgents.push_back(aT);
			}
		}
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

void Ped::Model::tick_OMP() {
	#pragma omp parallel for schedule(static) 
	for (int i = 0; i < agents.size(); i++) {
		Ped::Tagent* agent = agents[i]; 
		agent->computeNextDesiredPosition();
		agent->setX(agent->getDesiredX());
		agent->setY(agent->getDesiredY());
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
		// tick_OMP();;
		omp_set_num_threads(12);
		moveRegion();
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
	
	default:
		// for (auto agent : agents)
		// {
		// 	agent->computeNextDesiredPosition();
		// 	// agent->setX(agent->getDesiredX());
		// 	// agent->setY(agent->getDesiredY());
		// 	//move(agent);
		// }
		omp_set_num_threads(1);
		moveRegion();
		break;
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
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

void Ped::Model::moveRegion()
{
	for (size_t i = 0; i < regions.size(); i++)
	{
		regions.at(i).assignedAgents.clear();
	}
	for (size_t aT = 0; aT < agents.size(); aT++)
	{
		if (aT == 60)
		{
			// std::cout << "60 IN A REGION: " << aT << " " << agents.at(aT)->getX() << " " << agents.at(aT)->getY() << endl;
		}
		bool inR = false;
		for (size_t i = 0; i < regions.size(); i++)
		{
			if (agents.at(aT)->getX() >= regions.at(i).leftBorder && agents.at(aT)->getX() < regions.at(i).rightBorder) {
				regions.at(i).assignedAgents.push_back(aT);
				inR = true;
			}
		}
		if (!inR)
		{
			// std::cout << "NOT IN A REGION: " << aT << " " << agents.at(aT)->getX() << endl;
		}
		
	}
	// std::cout << "ITER" << endl;
	// std::cout << "R_SIZE " << regions.size() << " AGTENS: " << agents.size() << endl;
	#pragma omp parallel for
	for (size_t i = 0; i < regions.size(); i++)
	{
		for (size_t j = 0; j < regions.at(i).assignedAgents.size(); j++)
		{
			agents.at(regions.at(i).assignedAgents.at(j))->computeNextDesiredPosition();
			moveAgentRegion(regions.at(i).assignedAgents.at(j), i);
		}
	}

	#pragma omp parallel for
	for (size_t i = 0; i < regions.size(); i++)
	{
		// for (size_t j = 0; j < regions.at(i).assignedAgents.size(); j++)
		// {
		// 	agents.at(regions.at(i).assignedAgents.at(j))->computeNextDesiredPosition();
		// 	moveAgentRegion(regions.at(i).assignedAgents.at(j), i);
		// }
		for (size_t j = 0; j < regions.at(i).removeFromList.size(); j++)
		{
			// auto index = find(v.begin(), v.end(), 5) - v.begin();
			// regions.at(i).assignedAgents.erase(regions.at(i).assignedAgents.at(j));
			// Use std::remove to move the elements to be removed to the end
			auto newEnd = std::remove(regions.at(i).assignedAgents.begin(), regions.at(i).assignedAgents.end(), regions.at(i).removeFromList.at(j));

			// Erase the "removed" elements from the vector
			regions.at(i).assignedAgents.erase(newEnd, regions.at(i).assignedAgents.end());
		}
		regions.at(i).removeFromList.clear();
		for (size_t j = 0; j < REGION_Y_SIZE; j++)
		{
			int valueL = regions.at(i).leftBorderGates[j]->exchange(-1);
			if (valueL != -1)
			{
				regions.at(i).assignedAgents.push_back(valueL);
			}
			int valueR = regions.at(i).rightBorderGates[j]->exchange(-1);
			if (valueR != -1)
			{
				regions.at(i).assignedAgents.push_back(valueR);
			}
		}
	}

	// manageRegions();
}

void Ped::Model::manageRegions()
{
	// #pragma omp parallel for
	for (size_t i = 0; i < regions.size(); i++)
	{
		std::cout << i << ": " << regions.at(i).leftBorder << " " << regions.at(i).assignedAgents.size() << " " << regions.at(i).rightBorder << std::endl;
		if (regions.at(i).assignedAgents.size() > 50) {
			int lB = regions.at(i).leftBorder;
			int rB = regions.at(i).rightBorder;
			int nB = regions.at(i).leftBorder+(regions.at(i).rightBorder-regions.at(i).leftBorder)/2;
			regions.at(i).rightBorder = nB;

			Region regio;
			regio.leftBorder = nB;
			for (size_t j = 0; j < REGION_Y_SIZE; j++)
			{
				std::atomic<int>* atom = new std::atomic(-1);
				regio.leftBorderGates[j] = atom;
			}
			regio.rightBorder = rB;
			for (size_t j = 0; j < REGION_Y_SIZE; j++)
			{
				std::atomic<int>* atom = new std::atomic(-1);
				regio.rightBorderGates[j] = atom;
			}

			std::vector<int> oldVec = regions.at(i).assignedAgents;
			regions.at(i).assignedAgents.clear();
			for (size_t j = 0; j < oldVec.size(); j++)
			{
				if (agents.at(oldVec.at(j))->getX() <= nB)
				{
					regions.at(i).assignedAgents.push_back(oldVec.at(j));
				} else {
					regio.assignedAgents.push_back(oldVec.at(j));
				}
				
			}
			
			// regio.assignedAgents;

			regions.insert(regions.begin() + i+1, regio);
		}
	}

	// #pragma omp parallel for
	for (size_t i = 0; i < regions.size()-1; i++)
	{
		// std::cout << i << ": " << regions.at(i).leftBorder << " " << regions.at(i).assignedAgents.size() << " " << regions.at(i).rightBorder << std::endl;
		if (regions.at(i).assignedAgents.size()+regions.at(i+1).assignedAgents.size() < 50) {
			regions.at(i).rightBorder = regions.at(i+1).rightBorder;

			regions.at(i).assignedAgents.insert(regions.at(i).assignedAgents.end(), regions.at(i+1).assignedAgents.begin(), regions.at(i+1).assignedAgents.end());

			// regions.insert(regions.begin() + i+1, regio);
			regions.erase(regions.begin() + i+1);
		}
	}

	// for (size_t i = 0; i < regions.size(); i++)
	// {
	// 	std::cout << i << ": " << regions.at(i).leftBorder << " " << regions.at(i).assignedAgents.size() << " " << regions.at(i).rightBorder << std::endl;
	// }
}

void Ped::Model::moveAgentRegion(int aT, int regionIndex)
{
	// std::cout << "\tMOVEA" << endl;
	
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agents.at(aT)->getX(), agents.at(aT)->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agents.at(aT)->getDesiredX(), agents.at(aT)->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agents.at(aT)->getX();
	int diffY = pDesired.second - agents.at(aT)->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agents.at(aT)->getY());
		p2 = std::make_pair(agents.at(aT)->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	// std::cout << "\tMY" << endl;
	
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {
			// Set the agent's position 
			int lx = (*it).first;
			int ly = (*it).second;
			int32_t ex = -1;

			if (lx < regions.at(regionIndex).leftBorder && lx >= 0)
			{
				if (atomic_compare_exchange_strong(regions.at(regionIndex-1).rightBorderGates[ly], &ex, aT))
				{
					agents.at(aT)->setX((*it).first);
					agents.at(aT)->setY((*it).second);
					regions.at(regionIndex).removeFromList.push_back(aT);
					break;
				}
			}
			else if (lx >= regions.at(regionIndex).rightBorder && lx < REGION_X_SIZE)
			{
				if (atomic_compare_exchange_strong(regions.at(regionIndex+1).leftBorderGates[ly], &ex, aT))
				{
					agents.at(aT)->setX((*it).first);
					agents.at(aT)->setY((*it).second);
					regions.at(regionIndex).removeFromList.push_back(aT);
					break;
				}
			}
			else if (lx == regions.at(regionIndex).leftBorder && lx > 0)
			{
				if (atomic_compare_exchange_strong(regions.at(regionIndex).leftBorderGates[ly], &ex, aT))
				{
					agents.at(aT)->setX((*it).first);
					agents.at(aT)->setY((*it).second);
					regions.at(regionIndex).removeFromList.push_back(aT);
					break;
				}
			}
			else if (lx == regions.at(regionIndex).rightBorder-1 && lx <= REGION_X_SIZE)
			{
				if (atomic_compare_exchange_strong(regions.at(regionIndex).rightBorderGates[ly], &ex, aT))
				{
					agents.at(aT)->setX((*it).first);
					agents.at(aT)->setY((*it).second);
					regions.at(regionIndex).removeFromList.push_back(aT);
					break;
				}
			}
			else {
				agents.at(aT)->setX((*it).first);
				agents.at(aT)->setY((*it).second);
				break;
			}
		}
	}
	
	// bool retry = true;
	// while (retry)
	// {
	// 	if (regions.at(regionIndex).changeRegion->try_lock()) {
	// 		// std::cout << "\t\tL1" << endl;
	// 		int borderRegion = 0;
	// 		if (agents.at(aT)->getX() < regions.at(regionIndex).leftBorder+3 && agents.at(aT)->getX() > 0) {
	// 			borderRegion--;
	// 		} else if (agents.at(aT)->getX() > regions.at(regionIndex).rightBorder-3 && agents.at(aT)->getX() < 160) {
	// 			borderRegion++;
	// 		}
	// 		if (borderRegion == 0 || regions.at(regionIndex+borderRegion).changeRegion->try_lock()) {
	// 			// std::cout << "\t\tLOCKED IN" << endl;
	// 			retry = false;

	// 			for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {
			
	// 				// std::cout << "\t\t\tPRE-MOVED" << endl;
	// 				// If the current position is not yet taken by any neighbor
	// 				if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {
	// 					// Set the agent's position 
	// 					if (aT == 60)
	// 					{
	// 						// std::cout << "SET 60: " << (*it).first << endl;
	// 						// std::cout << "ALL ALT 60: " << prioritizedAlternatives.at(0).first << endl;
	// 						// std::cout << "ALL ALT 60: " << prioritizedAlternatives.at(1).first << endl;
	// 						// std::cout << "ALL ALT 60: " << prioritizedAlternatives.at(2).first << endl;
	// 					}

	// 					agents.at(aT)->setX((*it).first);
	// 					agents.at(aT)->setY((*it).second);
	// 					// std::cout << "\t\t\tMOVED" << endl;
	// 					break;
	// 				}
	// 			}
	// 			regions.at(regionIndex+borderRegion).changeRegion->unlock();
	// 		}
	// 		regions.at(regionIndex).changeRegion->unlock();
	// 	}
	// }
	
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
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
