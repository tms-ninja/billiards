#pragma once
#include <iostream>
#include "Vec2D.h"

enum class Collision_Type
{
	Disc_Disc,
	Disc_Wall,
	Disc_Boundary
};

class Event
{
public:
	Vec2D pos;
	Vec2D new_v;
	double t;
	size_t ind;			 // Index of the disc involved in the event
	size_t second_ind;   // index of the other thing involved in the event
	Collision_Type disc_wall_col;  // true if event is a disc-wall collision

	Event();

	Event(const double t, const size_t ind, const size_t second_ind, const Collision_Type disc_wall_col, const Vec2D &pos, const Vec2D &new_v);

	// Returns the index of the disc this disc is collising with. If it is a disc_wall collision
	// returns size_t_max
	size_t get_disc_partner() const;
};

std::ostream& operator<<(std::ostream &os, const Event &e);
