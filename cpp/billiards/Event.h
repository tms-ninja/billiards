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
	Vec2D pos;
	Vec2D new_v;
	double new_w;  // New angular velocity
	double t;
	size_t ind;			 // Index of the disc involved in the event
	size_t second_ind;   // index of the other thing involved in the event
	Collision_Type disc_wall_col;  // true if event is a disc-wall collision

	Event() = default;

	Event(const double t, const size_t ind, const size_t second_ind, const Collision_Type disc_wall_col, const Vec2D &pos, const Vec2D &new_v, const double new_w);

	// Returns the index of the disc this disc is collising with. If it is a disc_wall collision
	// returns size_t_max
	size_t get_disc_partner() const;
};

std::ostream& operator<<(std::ostream &os, const Event &e);
