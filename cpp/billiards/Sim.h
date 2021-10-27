#pragma once
#include <cmath>
#include <limits>
#include <vector>
#include "Vec2D.h"
#include "Disc.h"
#include "Wall.h"
#include "Event.h"

class Sim
{
public:
	std::vector<Disc> current_state;
	std::vector<Disc> initial_state;
	std::vector<Wall> walls;
	std::vector<Event> events;
	double current_time{ 0.0 };

	// Advances the simulation by either max_iterations or max_t, whichever is reached sooner
	void advance(size_t max_iterations, double max_t, bool record_events);

private:

	// Used for solving the quadratic involving the alpha and beta vectors
	// Returns infinity if there are no solutions, if there are solutions, it may still
	// return a negative value
	static double solve_quadratic(const Vec2D &alpha, const Vec2D &beta, const double R);

	// Tests if two discs are going to collide, returns time of collision if they do, infinity otherwise
	static double test_disc_disc_col(const Disc &d1, const Disc &d2);

	// Tests if a disc is going to collide with a wall, returns time of collision if they do, infinity otherwise
	static double test_disc_wall_col(const Disc &d, const Wall &w);

	// Performs a disc-wall collsi==ision
	static void disc_wall_col(Disc &d, const Wall &w);

	// performs a disc-wall collsi==ision
	static void disc_disc_col(Disc &d1, Disc &d2);

	// Computes position of disc d at time t
	static Vec2D disc_pos(const Disc& d, const double t);
};

