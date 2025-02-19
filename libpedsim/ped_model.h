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

#include <vector>
#include <map>
#include <set>
#include <atomic>

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
		void move(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		// All the regions
		std::vector<Region> regions;
		// each pixel of the space is an atomic boolean value
		std::vector<std::vector<std::atomic<bool>>> grid;

		

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
			std::vector<Tagent*> agents;
			std::vector<Region*> neighbors;
			
			Region(enum TYPE type, std::pair<int, int> min, std::pair<int, int> max) :type(type), min(min), max(max) {}
			~Region() {}

			// For a given region, initialize its adjacents
			// For each region in the space, check if any corner position of current region is in that region
			void initialize_neighbors(std::vector<Region> const &regions_in_space) {
				for (auto region: regions_in_space) {
					for (auto x: {min.first, min.second}) {
						for (auto y: {max.first, max.second}) {
							if (pos_in_region({x, y}) && &region != this) {
								neighbors.push_back(&region);	
							}
						}
					}
				}
			}

			// Record all the agents in this region at the beginning
			void initialize_agents(std::vector<Tagent*> const &agents) {
				for (auto agent: agents) {
					if (pos_in_region({agent->getX(), agent->getY()}) && !agent->is_owned) {
						this->agents.push_back(agent);
						agent->is_owned = true;
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
	};
}
#endif
