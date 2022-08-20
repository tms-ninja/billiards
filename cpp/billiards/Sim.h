// billiards: Program to model collisions between 2d discs
// Copyright (C) 2022  Tom Spencer (tspencerprog@gmail.com)
//
// This file is part of billiards
//
// billiards is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "Vec2D.h"
#include "Disc.h"
#include "Wall.h"
#include "Event.h"

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr size_t size_t_max = std::numeric_limits<size_t>::max();

class Sim
{
public:
	std::vector<Disc> initial_state;

	// bottom left and top right corners of the box the simulation takes place in
	const Vec2D bottom_left;
	const Vec2D top_right;
	std::vector<Wall> walls;  // Walls containing the simulation

	// Sectors
	// Horizontal and vertical number of sectors, this is the true number including 
	// the "shell" of sectors outside the simulation box
	const size_t N, M;

	// Width and height of each sector
	const double sector_width, sector_height;
	
	// Contains the events that occured during the simulation
	std::vector<Event> events;

	std::vector<size_t> new_vec, old_vec;
	std::vector<std::array<Event, 2>> events_vec;

	double current_time{ 0.0 };

	// N and M are the number of sectors the simulation should be split up into in the 
	// horizontal and vertical directions respectively
	Sim(Vec2D bottom_left, Vec2D top_right, size_t N=1, size_t M=1);

	// Advances the simulation by either max_iterations or max_t, whichever is reached sooner
	void advance(size_t max_iterations, double max_t, bool record_events);

	// Sets up the Sim instance ready for advance() to be called. Effectively just copies
	// current_state to initial_state so we know how it began
	void setup();

	// Adds a disc to the simulation. Should not be called after the simulation has started
	void add_disc(const Vec2D& pos, const Vec2D& v, double w, double m, double R, double I);

	// Getter/setter methods for coefficients of restitution

	double get_e_n() const;
	void set_e_n(double new_e_n);

	double get_e_t() const;
	void set_e_t(double new_e_t);

	Vec2D get_g() const;
	void set_g(const Vec2D& new_g);

private:
	double e_n{ 1.0 };   // Coefficient of normal restitution, default is perfectly elasic collisions
	double e_t{ -1.0 };  // Coefficient of tangential restitution, default is perfectly smooth

	// Acceleration due to gravity, default is zero
	Vec2D g{ 0.0, 0.0 };

	// First N+1 horizontal boundaries from left to right, then M+1 vertical bounraries from 
	// bottom to top
	std::vector<Wall> boundaries;

	// Keeps track of which balls are in which sector
	std::vector<std::vector<size_t>> sector_entires;

	// Coordinates of the centre of each sector
	std::vector<Vec2D> sector_centres;


	// Binary heap that minimises time[disc_ind, new[disc_ind]], so 
	// time[pth[disc_ind], new[pth[disc_ind]]] is minimum
	// **pth should always take a index for the heap, i.e. returns the ith disc
	// in the heap**
	std::vector<size_t> pth;

	// Inverse mapping giving posiiton of a disc in the heap, disc disc_ind is 
	// at position pht[disc_ind] in pth
	// **pht should always take the index of a disc**
	std::vector<size_t> pht;

	// gets the current time for a disc, time[disc_ind, new[disc_ind]]
	double get_time(size_t disc_ind) const;

	// Returns the index of of the disc whose time is smallest
	size_t get_min_time_ind() const;

	// Adds the time corresponding to disc_ind in initial_state to the heap
	void add_time_to_heap(size_t disc_ind);

	// Down heapify the element in the heap with disc index disc_ind
	void down_heapify(size_t disc_ind);

	// Up heapify the element in the heap with disc index disc_ind
	void up_heapify(size_t disc_ind);

	// Updates the time of disc ind with new_t
	void update_time(size_t disc_ind, double new_t);


	// Returns the Wall index which has the next collision for the disc and time
	std::pair<size_t, double> get_next_wall_coll(size_t disc_ind) const;

	// Returns the boundary index which has the next collision for the disc and time
	std::pair<size_t, double> get_next_boundary_coll(size_t disc_ind) const;

	// Returns the disc index which has the next collision for the disc and time
	std::pair<size_t, double> get_next_disc_coll(size_t disc_ind) const;


	// Used for solving the quadratic involving the alpha and beta vectors
	// Returns infinity if there are no solutions, if there are solutions, it may still
	// return a negative value
	static double solve_quadratic(const Vec2D &alpha, const Vec2D &beta, const double R);

	// Tests if two discs are going to collide, returns time of collision if they do, infinity otherwise
	double test_disc_disc_col(const Disc &d1, const Disc &d2, const Event &e1, const Event &e2) const;

	// Tests if a disc is going to collide with a wall, returns time of collision if they do, infinity otherwise
	// Differs from test_disc_boundary_col() as it tests for the interaction of the edge of the disc with the 
	// wall rather than the centre
	double test_disc_wall_col(const Disc& d, const Event &e, const Wall &w) const;

	// Tests if a disc is going to collide with a boundary. Differs from test_disc_wall_col() as it tests 
	// for the interaction of the centre of the disc with the boundary rather than the edge
	double test_disc_boundary_col(const Event& e, const Wall& w) const;

	// Performs a disc-wall collision
	void disc_wall_col(const Disc& d, Event &e, const Wall &w);

	// performs a disc-disc collision
	void disc_disc_col(Disc &d1, Disc &d2, Event& e1, Event &e2);

	// Advances the disc to time t
	void advance(const Event &old_e, Event &new_e, double t);

	// Computes the position of a particle with initial position pos, velocity v after a time dt
	Vec2D advance_position(const Vec2D& pos, const Vec2D& v, double dt) const;

	// Computes the velocity of a particle with initial position pos, velocity v after a time dt
	Vec2D advance_velocity(const Vec2D& pos, const Vec2D& v, double dt) const;

	// Computes the sector ID given its coordinates
	inline size_t sector_coords_to_ID(size_t x, size_t y) const;

	// Computes the sector's Cartesian corrdinates from its sector ID
	inline std::pair<size_t, size_t> sector_ID_to_coords(size_t sector_ID) const;

	// Returns the sector ID of the sector the given position is in
	size_t compute_sector_ID(const Vec2D& pos) const;

	// Updates the sector ID of disc with index disc_ind based on the old_event that has just been processed
	// Intended to be used after swapping new and old in advance()
	void update_sector_ID(const size_t disc_ind, const Event& old_event);

	// Checks whether the disc is leaving the current sector listed as its sector_ID
	// Returns true if it is leaving, false otherwise
	// Used to help determine whether we have or haven't processed a boundary interaction
	// when a disc is on/near a boundary
	// v should be the velocity of the disc when it is at the boundary
	bool check_leaving_sector(const Vec2D& v, const Wall& b, size_t disc_ind) const;

	// Verifies the sector ID of the disc places it within the bounds of the simulation
	void verify_disc_within_bounds(size_t disc_ind) const;
};

