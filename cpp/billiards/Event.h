#pragma once
#include "Vec2D.h"

class Event
{
public:
	Vec2D pos;
	Vec2D new_v;
	double t;
	size_t ind;

	Event(const Vec2D &pos, const Vec2D &new_v, const double t, const size_t ind);
};

