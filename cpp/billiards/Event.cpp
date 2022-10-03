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

Event::Event(const double t, const size_t ind, const size_t partner_ind, const Collision_Type col_type, const Vec2D& r, const Vec2D& v, const double w)
	: r{ r }, v{ v }, w{ w }, t{ t }, ind{ ind }, partner_ind{ partner_ind }, col_type{ col_type }
{
}

size_t Event::get_disc_partner() const
{
	if (col_type == Collision_Type::Disc_Disc)
		return partner_ind;
	else
		return std::numeric_limits<size_t>::max();
}

std::ostream & operator<<(std::ostream & os, const Event & e)
{
	os << e.t << '\t' << e.ind << '\t' << e.partner_ind << '\t' << static_cast<int>(e.col_type) << '\t' << e.r << '\t' << e.v << '\t' << e.w;

	return os;
}
