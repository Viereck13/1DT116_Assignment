//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <unordered_map>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <atomic>
#include <assert.h>
#include <iostream>

#include "ped_agent.h"

namespace Ped{
	class Tagent;
	class Region;
	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };
	enum TYPE {TYPE1, TYPE2, TYPE3};

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// Returns the agents of this scenario
		const std::vector<Tagent*>& getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);  // original move method 
		void move_trivial(Ped::Tagent *agent, Ped::Region &region);  // my trivial move method
    std::vector<std::pair<int, int>> find_alternatives(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		// All the regions
		std::vector<Region> regions;
		// each pixel of the space is an atomic boolean value
    std::atomic<bool> **grid;

  Ped::Region* get_target_region(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
	};

	class Region {
		public:
			enum TYPE type;
			std::pair<int, int> min, max;	// (x_min, y_min) and (x_max, y_max) can define a rectangle region	
			std::unordered_set<Tagent*> agents;
			std::vector<Region*> neighbors;
      
      std::atomic<bool> *in_use;
      std::deque<Tagent*> inqueue;
      std::deque<Tagent*> outqueue;
			
			Region(enum TYPE type, std::pair<int, int> min, std::pair<int, int> max) :type(type), min(min), max(max){
        in_use[0].store(false, std::memory_order_relaxed);
        in_use = (std::atomic<bool>*)malloc(sizeof(std::atomic<bool>));
        in_use->store(false, std::memory_order_relaxed);
      }

			// For a given region, initialize its adjacents
			// For each region in the space, check if any corner position of current region is in that region
			void initialize_neighbors(std::vector<Region> &regions_in_space) {
        int flag = 0; // When a region detects a neighbor region, can skip detection for the same neighbor region
				for (auto &region: regions_in_space) {
          flag = 0;
					for (auto x: {min.first, max.first}) {
						for (auto y: {min.second, max.second}) {
							if (region.pos_in_region({x, y}) && &region != this) {
                if (flag) break;
                flag = 1;
								neighbors.push_back(&region);	
							}
						}
					}
				}
			}

			// Record all the agents in this region at the beginning
			void initialize_agents(std::vector<Tagent*> &agents) {
				for (auto agent: agents) {
					if (pos_in_region({agent->getX(), agent->getY()}) && !agent->is_owned) {
						agent->is_owned = true;
						this->agents.insert(agent);
					}
				}
			}

			// Check if a position is in this region
			bool pos_in_region(std::pair<int, int> pos) {
				return pos.first >= min.first
					&& pos.first <= max.first
					&& pos.second >= min.second
					&& pos.second <= max.second;
			}

      // Each time at the beginning of the tick, update agents in every region
      void update_agents() {
        // remove agents which are not in this region any more
        while (!outqueue.empty()) {
          agents.erase(outqueue.front());
          outqueue.pop_front();
        }
        // insert new agents
        bool expected = false;
        bool new_value = true;
        while (!in_use->compare_exchange_strong(expected, new_value)) {}
        while (!inqueue.empty()) {
          agents.insert(inqueue.front());  
          inqueue.pop_front();
        }
        in_use->store(false);
      }
	};
}
#endif
