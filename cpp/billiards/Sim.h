#pragma once
#include <array>
#include <cmath>
#include <limits>
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
	//std::vector<Disc> current_state;
	std::vector<Disc> initial_state;
	std::vector<Wall> walls;
	std::vector<Event> events;

	std::vector<size_t> new_vec, old_vec;
	std::vector<std::array<Event, 2>> events_vec;

	double current_time{ 0.0 };

	// Advances the simulation by either max_iterations or max_t, whichever is reached sooner
	void advance(size_t max_iterations, double max_t, bool record_events);

	// Sets up the Sim instance ready for advance() to be called. Effectively just copies
	// current_state to initial_state so we know how it began
	void setup();

private:
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
	double get_time(size_t disc_ind);

	// Returns the Disc whose time is smallest
	size_t get_min_time_ind();

	// Adds the time corresponding to disc_ind to the heap
	void add_time_to_heap(size_t disc_ind);

	// Removes the time corresponding to disc_ind from the heap
	void down_heapify(size_t disc_ind);

	void up_heapify(size_t disc_ind);


	// Updates the time of disc ind
	void update_time(size_t disc_ind, double new_t);

	void verify_heap();

	// Returns the Wall index which has the next collision for the disc and time
	std::pair<size_t, double> get_next_wall_coll(size_t disc_ind);

	// Returns the disc index which has the next collision for the disc and time
	std::pair<size_t, double> get_next_disc_coll(size_t disc_ind);


	// Used for solving the quadratic involving the alpha and beta vectors
	// Returns infinity if there are no solutions, if there are solutions, it may still
	// return a negative value
	static double solve_quadratic(const Vec2D &alpha, const Vec2D &beta, const double R);

	// Tests if two discs are going to collide, returns time of collision if they do, infinity otherwise
	static double test_disc_disc_col(const Disc &d1, const Disc &d2, const Event &e1, const Event &e2);

	// Tests if a disc is going to collide with a wall, returns time of collision if they do, infinity otherwise
	static double test_disc_wall_col(const Disc& d, const Event &e, const Wall &w);

	// Performs a disc-wall collision
	static void disc_wall_col(Event &e, const Wall &w);

	// performs a disc-wall collision
	static void disc_disc_col(Disc &d1, Disc &d2, Event& e1, Event &e2);

	// Computes position of disc d at time t
	static Vec2D disc_pos(const Disc& d, const double t);

	// Advances the disc to time t
	static void advance(const Event &old_e, Event &new_e, double t);
};

