#pragma once
#include <iostream>
#include "Vec2D.h"

class Event
{
public:
	Vec2D pos;
	Vec2D new_v;
	double t;
	size_t ind;			 // Index of the disc involved in the event
	size_t second_ind;   // index of the other thing involved in the event
	bool disc_wall_col;  // true if event is a disc-wall collision

	Event();

	Event(const double t, const size_t ind, const size_t second_ind, const bool disc_wall_col, const Vec2D &pos, const Vec2D &new_v);
};

std::ostream& operator<<(std::ostream &os, const Event &e);
