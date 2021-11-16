#include "Event.h"

Event::Event()
{
}

Event::Event(const double t, const size_t ind, const size_t second_ind, const bool disc_wall_col, const Vec2D &pos, const Vec2D &new_v)
	: pos{ pos }, new_v{ new_v }, t{ t }, ind{ ind }, second_ind{ second_ind }, disc_wall_col{ disc_wall_col }
{
}

size_t Event::get_disc_partner() const
{
	return disc_wall_col ? std::numeric_limits<size_t>::max() : second_ind;
}

std::ostream & operator<<(std::ostream & os, const Event & e)
{
	std::cout << e.t << '\t' << e.ind << '\t' << e.second_ind << '\t' << e.disc_wall_col << '\t' << e.pos << '\t' << e.new_v;

	return os;
}
