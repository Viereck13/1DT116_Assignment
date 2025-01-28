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

#ifndef NOCDUA
#include "cuda_testkernel.h"
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

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::tick()
{
	// EDIT HERE FOR ASSIGNMENT 1
	// std::cout << "The length of the vector is: " << agents.size() << std::endl;
    // for (auto agent : agents)
    // {
    //     // Find the next position for the agent
    //     agent->computeNextDesiredPosition();

    //     // Set the agent's position to that position
    //     agent->setX(agent->getDesiredX());
    //     agent->setY(agent->getDesiredY());
    // }
	// ----------------------------------------------
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < agents.size(); i++) {
		Ped::Tagent* agent = agents[i]; 
		agent->computeNextDesiredPosition();
		// printf("Thread %d is processing agent %d\n", omp_get_thread_num(), i);
		
		agent->setX(agent->getDesiredX());
		agent->setY(agent->getDesiredY());
	}
	// ----------------------------------------------
	// std::vector<std::thread> threads;

	// for (auto agent : agents) {
	// 	threads.emplace_back([agent]() {
	// 		agent->computeNextDesiredPosition();
	// 		agent->setX(agent->getDesiredX());
	// 		agent->setY(agent->getDesiredY());
	// 	});
	// }

    // // Join threads
    // for (std::thread& t : threads) {
    //     if (t.joinable()) {
    //         t.join();
    //     }
    // }

	// 1 thread per agent
	// The number of threads can be too high, so it has too much overhead.
	// ----------------------------------------------
	// Many agents per thread, distributed equally
	// Limit the number of threads to the number of (logical) cores 

    // std::vector<std::thread> threads;

    // // Lambda function to update a subset of agents
    // auto updateAgents = [](std::vector<Ped::Tagent*>::iterator start, std::vector<Ped::Tagent*>::iterator end) {
    //     for (auto it = start; it != end; it++) {
    //         Ped::Tagent* agent = *it;

    //         agent->computeNextDesiredPosition();
    //         agent->setX(agent->getDesiredX());
    //         agent->setY(agent->getDesiredY());
    //     }
    // };

    // // Determine the number of agents per thread, and the number of threads will be limited by the hardware.
    // const unsigned int numThreads = std::thread::hardware_concurrency(); // Returns the number of concurrent threads supported by the implementation.
    // const unsigned int agentsPerThread = agents.size() / numThreads;
	// // cout << "Number of threads: " << numThreads << endl;
	// // cout << "Number of agents: " << agents.size() << endl;
	// // cout << "Agents per thread: " << agentsPerThread << endl;
    

	// auto agentStart = agents.begin();
    // for (int i = 0; i < numThreads; i++) {
    //     auto agentEnd = (i == numThreads-1) ? agents.end() : agentStart + agentsPerThread;
    //     threads.emplace_back(updateAgents, agentStart, agentEnd);
    //     agentStart = agentEnd; // The first agent of the next thread is the last agent of the current thread
    // }

    // // Join threads
    // for (std::thread& t : threads) {
    //     if (t.joinable()) {
    //         t.join();
    //     }
    // }
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
