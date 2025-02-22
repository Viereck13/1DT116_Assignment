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
#include <atomic>
#include <algorithm>
#include <omp.h>
#include <utility>
#include <cmath>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif



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

  regions.push_back(new Region(TYPE1, {0, 0}, {40, 120}));
  regions.push_back(new Region(TYPE1, {40, 0}, {80, 120}));
  regions.push_back(new Region(TYPE1, {80, 0}, {120, 120}));
  regions.push_back(new Region(TYPE1, {120, 0}, {160, 120}));

  for (auto &region: regions) {
		region->initialize_neighbors(regions);
		region->initialize_agents(agents);
    /*std::cout << "region has : " << region.agents.size() << std::endl;*/
    /*std::cout << "region has : " << region.neighbors.size() << std::endl;*/
	}

  // Initialize cas array
  {
    int row = 121;
    int col = 161;
    grid = (std::atomic<bool> **)malloc(row * sizeof(void*));

    for (int i = 0; i < row; i++)
      grid[i] = (std::atomic<bool> *)malloc(col * sizeof(std::atomic<bool>));

    for (int i = 0; i < row; i++)
      for (int j = 0; j < col; j++)
        grid[i][j].store(false, std::memory_order_relaxed);

    for (auto const &agent: agents)
      grid[agent->getY()][agent->getX()].store(true);
  }

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::tick()
{
  switch (implementation) {
    case SEQ:
	    for (int i = 0; i < agents.size(); i++) {
        agents[i]->computeNextDesiredPosition();
        move(agents[i]);
      }
      break;
    case OMP:
      // Reallocate agents into different regions after each tick
      for (auto agent: agents) {
        agent->is_owned = false;
      }
      for (auto region: regions) {
        region->agents.clear();
        region->initialize_agents(agents);
      }
      #pragma omp parallel
      {
        auto tid = omp_get_thread_num();
        if (tid < regions.size()) {
          // thread tid is responsible for regions[tid]
          Ped::Region *region = regions[tid];
          for (auto agent: region->agents)	{
            agent->computeNextDesiredPosition();
            move_trivial(agent, *region);
          }
        }
      }
      break;
    case CUDA:
    case VECTOR:
    case PTHREAD:
		  break;
  }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////
std::vector<std::pair<int, int>> Ped::Model::find_alternatives(Ped::Tagent *agent) {
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

  return std::move(prioritizedAlternatives);
}

void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve neighbors' positions, then push each neighbor's position into 'takenPositions' pair
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives = find_alternatives(agent);
	

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

void Ped::Model::move_trivial(Ped::Tagent *agent, Ped::Region &region)
{
	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives = find_alternatives(agent);
	
	bool expected = false;
	bool new_value = true;
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {
    // Two cases: agents in the inner region or agents near border
    if (dest_around_border(*it)) {
      if (grid[(*it).second][(*it).first].compare_exchange_strong(expected, new_value)) {
        grid[agent->getY()][agent->getX()].store(false);
        agent->setX((*it).first);
        agent->setY((*it).second);
			  break;
      }

      /*// If after updating this agent is no longer in current region*/
      /*if (!region.pos_in_region({agent->getX(), agent->getY()})) {*/
      /*  region.outqueue.push_back(agent);*/
      /**/
      /*  // Where this agent goes*/
      /*  Ped::Region *region = get_target_region(agent);*/
      /*  while (!region->in_use->compare_exchange_strong(expected, new_value)) {}*/
      /*    region->inqueue.push_back(agent); */
      /*}*/
      
		} else {
      grid[agent->getY()][agent->getX()].store(false);
      agent->setX((*it).first);
      agent->setY((*it).second);
      break;
    }
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// At now, it just returns all agents in the space, because now we only have a unique region.
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

/*Ped::Region* Ped::Model::get_target_region(Ped::Tagent *agent)	{*/
/*  Ped::Region *res = NULL;*/
/*  int x = agent->getX();*/
/*  int y = agent->getY();*/
/*  for (int i = 0; i < regions.size(); i++) {*/
/*    if (regions[i].pos_in_region({x, y})) {*/
/*      res = &regions[i];*/
/*    }*/
/*  }*/
/**/
/*  assert(res != NULL);*/
/*  return res;*/
/*}*/

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
