#pragma once
#include <iostream>
#include "Vec2D.h"

class Event
{
public:
	Vec2D pos;
	Vec2D new_v;
	double t;
	size_t ind;

	Event(const double t, const size_t ind, const Vec2D &pos, const Vec2D &new_v);
};

std::ostream& operator<<(std::ostream &os, const Event &e);
