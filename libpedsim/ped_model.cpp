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

	////// Collect all agents and distribute info to vectors
	for (auto agent : agents)
	{
		x.push_back(agent->getX());
		y.push_back(agent->getY());
		// desiredX.push_back(agent->getDesiredX());
		// desiredY.push_back(agent->getDesiredY());
		// printf("LOL\n");
		// if (agent->destination == NULL) {printf("R.I.P\n");};
		Twaypoint* temp = agent->waypoints.front();
		destinationX.push_back(temp->getx());
		destinationY.push_back(temp->gety());
		destinationR.push_back(temp->getr());
		agent->destination = temp;
		agent->waypoints.pop_front();
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
		int agentReachedDestination = _mm256_testnzc_si256(compare,_mm256_set1_epi64x(1));

		// check if min one agent needs new destination
		if (agentReachedDestination) {
			// printf("IFIFIFIFIF\n");
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
	
		// write back to agents
		for (size_t j = 0; j < 4; j++)
		{
			if (i+j < agents.size())
			{
				agents.at(i+j)->x = x.at(i+j);
				agents.at(i+j)->desiredPositionX = x.at(i+j);
				agents.at(i+j)->y = y.at(i+j);
				agents.at(i+j)->desiredPositionY = x.at(i+j);
			}
		}
	}
}

void Ped::Model::tick()
{
	tick_SIMD();
	// EDIT HERE FOR ASSIGNMENT 1
	// for (auto agent : agents)
    // {
    //     agent->computeNextDesiredPosition();
    //     agent->setX(agent->getDesiredX());
    //     agent->setY(agent->getDesiredY());
    // }
	// ----------------------------------------------------
	// #pragma omp parallel for schedule(static) 
	// for (int i = 0; i < agents.size(); i++) {
	// 	Ped::Tagent* agent = agents[i]; 
	// 	agent->computeNextDesiredPosition();
	// 	// printf("Thread %d is processing agent %d\n", omp_get_thread_num(), i);
		
	// 	agent->setX(agent->getDesiredX());
	// 	agent->setY(agent->getDesiredY());
	// }

	// -------------C++ threads version--------------------
  // size_t threads = 8; 
  // std::vector<std::thread> handles;
  // size_t quotient = agents.size() / threads;
  // size_t remainder = agents.size() % threads;
  // for (int i = 0; i < threads; i++) {
  //   handles.push_back(std::thread([this](size_t begin, size_t length) {
  //     for (int i = begin; i < begin + length; i++) {
  //       agents[i]->computeNextDesiredPosition();
  //       agents[i]->setX(agents[i]->getDesiredX());
  //       agents[i]->setY(agents[i]->getDesiredY());
  //     }
  //   }, i*quotient, i==threads-1 ? quotient+remainder : quotient));
  // }
  //
  // for (auto& t: handles)
  //   if (t.joinable())
  //     t.join();
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
