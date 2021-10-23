#include "Event.h"

Event::Event(const double t, const size_t ind, const Vec2D &pos, const Vec2D &new_v)
	: pos{ pos }, new_v{ new_v }, t{ t }, ind{ ind }
{
}

std::ostream & operator<<(std::ostream & os, const Event & e)
{
	std::cout << e.t << '\t' << e.ind << '\t' << e.pos << '\t' << e.new_v;

	return os;
}
