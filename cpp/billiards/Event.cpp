#include "Event.h"

Event::Event(const Vec2D & pos, const Vec2D & new_v, const double t, const size_t ind)
	: pos{ pos }, new_v{ new_v }, t{ t }, ind{ ind }
{
}
