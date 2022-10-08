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

#pragma once
#include "Vec2D.h"

enum class Collision_Type
{
	Disc_Disc,
	Disc_Wall,
	Disc_Boundary,
	Disc_Advancement
};

class Event
{
public:
	Vec2D r;
	Vec2D v;
	double w;  // New angular velocity
	double t;
	size_t ind;			 // Index of the disc involved in the event
	size_t partner_ind;   // index of the other thing involved in the event
	Collision_Type col_type;

	Event() = default;

	Event(const double t, const size_t ind, const size_t partner_ind, const Collision_Type col_type, const Vec2D &r, const Vec2D &v, const double w);

	// Returns the index of the disc this disc is collising with. If it is a disc_wall collision
	// returns size_t_max
	size_t get_disc_partner() const;
};

std::ostream& operator<<(std::ostream &os, const Event &e);
