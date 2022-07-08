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

#include "Event.h"

Event::Event(const double t, const size_t ind, const size_t second_ind, const Collision_Type disc_wall_col, const Vec2D &pos, const Vec2D &new_v)
	: pos{ pos }, new_v{ new_v }, t{ t }, ind{ ind }, second_ind{ second_ind }, disc_wall_col{ disc_wall_col }
{
}

size_t Event::get_disc_partner() const
{
	if (disc_wall_col == Collision_Type::Disc_Disc)
		return second_ind;
	else
		return std::numeric_limits<size_t>::max();
}

std::ostream & operator<<(std::ostream & os, const Event & e)
{
	os << e.t << '\t' << e.ind << '\t' << e.second_ind << '\t' << static_cast<int>(e.disc_wall_col) << '\t' << e.pos << '\t' << e.new_v;

	return os;
}
