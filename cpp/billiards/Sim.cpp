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

#include "Sim.h"

Sim::Sim(Vec2D bottom_left, Vec2D top_right, size_t N, size_t M) 
	: bottom_left{bottom_left}, top_right{top_right}, N{N+2}, M{M+2},
	sector_width{ (top_right[0] - bottom_left[0]) / N },
	sector_height{ (top_right[1] - bottom_left[1]) / M },
	sector_entires((N+2)*(M+2))
{
	// Validate number of sectors to be used
	if (N < 1)
		throw std::invalid_argument("Number of sectors in x direction must be at least 1");
	else if (M < 1)
		throw std::invalid_argument("Number of sectors in y direction must be at least 1");

	// Add walls as we have the dimensions of the box
	double left, right, top, bottom;

	left = bottom_left[0];
	bottom = bottom_left[1];
	right = top_right[0];
	top = top_right[1];

	if (right <= left)
		throw std::invalid_argument("Top-right corner of simulation box cannot be to the left of bottom-left corner");
	else if (top <= bottom)
		throw std::invalid_argument("Top-right corner of simulation box cannot be below bottom-left corner");

	walls.push_back(Wall{ { left,  bottom }, { left,  top } });
	walls.push_back(Wall{ { left,  top },    { right, top } });
	walls.push_back(Wall{ { right, top },    { right, bottom } });
	walls.push_back(Wall{ { right, bottom }, { left,  bottom } });

	// Add horizontal boundaires first, then vertical
	for (int i = 0; i <= this->N; i++)
	{
		double x_pos{ left + (i-1) * sector_width };

		boundaries.push_back(Wall{ {x_pos, top + sector_height }, {x_pos, bottom - sector_height} });
	}

	for (int i = 0; i <= this->M; i++)
	{
		double y_pos{ bottom + (i-1) * sector_height };

		boundaries.push_back(Wall{ {left - sector_width, y_pos}, {right + sector_width, y_pos} });
	}
}

void Sim::advance(size_t max_iterations, double max_t, bool record_events)
{
	// Implements the algorithm described in (Lubachevsky 1991)

	size_t current_it{ 0 };
	double current_t{ this->current_time };

	// true means we are still finding disc with their new event time of 0.0
	// Collisions for discs may not be performed in the order they take place
	// during this time, so we shouldn't count these as interactions as they may
	// be undone
	bool initial_setup{ true }; 

	size_t i, j, k, m, boundary_ind;
	double P, Q, R, boundary_t;

	// Run to max_iterations*2 as disc-disc collisions count involve two iterations
	// Similarly, need to make disc-wall collisions count as 2
	while (current_it < 2*max_iterations && current_t < max_t)
	{
		// Determine i
		i = get_min_time_ind();
		current_t = get_time(i);

		if (current_t > 0.0)
			initial_setup = false;

		// Swap new and old for disc i
		new_vec[i] = old_vec[i];
		old_vec[i] = 1 - new_vec[i];

		Event& old_event_i{ events_vec[i][old_vec[i]] };

		// update sector IDs
		update_sector_ID(i, old_event_i);

		// Update count of how many collisions have been performed
		// We count disc-disc as one collision so we count disc-wall as 2 to avoid double counting
		// disc-disc collisions 
		if (!initial_setup)
			if (old_event_i.disc_wall_col==Collision_Type::Disc_Wall)
				current_it += 2;
			else if (old_event_i.disc_wall_col == Collision_Type::Disc_Disc)
				current_it += 1;

		// Save old state as that is the event that has just been processed
		if (!initial_setup && record_events && 
			(
				old_event_i.disc_wall_col == Collision_Type::Disc_Disc ||
				old_event_i.disc_wall_col == Collision_Type::Disc_Wall)
			)
			events.push_back(old_event_i);

		// Minimise over other discs
		std::tie(j, P) = get_next_disc_coll(i);

		// Minimise over walls
		std::tie(k, Q) = get_next_wall_coll(i);

		// Minimise over boundaries
		std::tie(boundary_ind, boundary_t) = get_next_boundary_coll(i);

		R = std::min(std::min(Q, boundary_t), P);

		update_time(i, R);

		// Set about advancing discs and performing collisions
		if (R < infinity)
		{
			Event& state_1{ events_vec[i][new_vec[i]] };

			Sim::advance(old_event_i, state_1, R);

			if (boundary_t < P || Q < P)  // Collision with either wall or boundary first
			{
				if (boundary_t < Q)  // Collision with boundary first
				{
					state_1.second_ind = boundary_ind;
					state_1.disc_wall_col = Collision_Type::Disc_Boundary;
				}
				else  // Collision with wall first
				{
					Sim::disc_wall_col(state_1, walls[k]);

					state_1.second_ind = k;
					state_1.disc_wall_col = Collision_Type::Disc_Wall;
				}
			}
			else  // Disc-disc collision
			{
				Event& state_2{ events_vec[j][new_vec[j]] };

				update_time(j, R);

				Sim::advance(events_vec[j][old_vec[j]], state_2, R);

				Sim::disc_disc_col(initial_state[i], initial_state[j], state_1, state_2);

				m = state_2.get_disc_partner();

				state_1.second_ind = j;
				state_2.second_ind = i;

				state_1.disc_wall_col = Collision_Type::Disc_Disc;
				state_2.disc_wall_col = Collision_Type::Disc_Disc;

				if (m != size_t_max && m != i)
				{
					Event& state_m{ events_vec[m][new_vec[m]] };

					Sim::advance(events_vec[m][old_vec[m]], state_m, state_m.t);

					state_m.second_ind = size_t_max;
					state_m.disc_wall_col = Collision_Type::Disc_Advancement;
				}
			}
		}
	}
}

void Sim::setup()
{
	events_vec.resize(initial_state.size());

	new_vec.resize(initial_state.size(), 0);
	old_vec.resize(initial_state.size(), 1);

	pht.resize(initial_state.size());

	for (size_t ind = 0; ind < initial_state.size(); ++ind)
	{
		auto& elem{ events_vec[ind] };
		Disc& disc{ initial_state[ind] };

		elem[0].t = 0.0;
		elem[0].pos = disc.r;
		elem[0].new_v = disc.v;
		elem[0].ind = ind;
		elem[0].second_ind = std::numeric_limits<size_t>::max();
		disc.sector_ID = compute_sector_ID(disc.r);
		
		elem[1] = elem[0];

		// Add to the heap
		add_time_to_heap(ind);
	}
}

void Sim::add_disc(const Vec2D& pos, const Vec2D& v, double m, double R)
{
	// Check disc is within simulation bounds
	double left, right, top, bottom;

	left = bottom_left[0];
	bottom = bottom_left[1];
	right = top_right[0];
	top = top_right[1];

	if (
		pos[0]-R < left ||
		pos[0]+R > right ||
		pos[1]-R < bottom ||
		pos[1]+R > top
	)
		throw std::invalid_argument("Can't add disc outside the simulation");

	// Check m is greater than zero
	if (m <= 0.0)
		throw std::invalid_argument("Can't add disc with mass less than or equal to zero");

	// check R is greater than zero
	if (R <= 0.0)
		throw std::invalid_argument("Can't add disc with radius less than or equal to zero");

	// check diameter isn't less than sector size
	if (2.0 * R >= sector_width || 2.0*R >= sector_height)
		throw std::invalid_argument("Can't add disc with radius greater than or equal to sector width/height");

	size_t sector_ID{compute_sector_ID(pos)};

	sector_entires.at(sector_ID).push_back(initial_state.size());

	initial_state.emplace_back(pos, v, m, R, sector_ID);
}

double Sim::get_time(size_t disc_ind)
{
	return events_vec[disc_ind][new_vec[disc_ind]].t;
}

size_t Sim::get_min_time_ind()
{
	// index of disc with smallest time is at root of heap pht
	return pth[0];
}

void Sim::add_time_to_heap(size_t disc_ind)
{
	pth.push_back(disc_ind);
	
	// Assume pht is of correct size
	pht[disc_ind] = pth.size() - 1;

	up_heapify(disc_ind);
}

void Sim::down_heapify(size_t disc_ind)
{
	// parent, child_1 & child_2 are indices in pht, i.e. pht[parent]
	size_t parent{ pht[disc_ind] };
	size_t child_1, child_2;

	while (true)
	{
		child_1 = 2 * parent + 1;
		child_2 = 2 * parent + 2;

		if (child_1 > pth.size() - 1)
		{
			// No children
			break;
		}
		else if (child_2 > pth.size() - 1)
		{
			// Only one child, child 1
			if (get_time(pth[child_1]) < get_time(pth[parent]))
			{
				// Need to swap them in pht
				std::swap(pht[pth[parent]], pht[pth[child_1]]);

				// Need to swap parent and current in heap pth
				std::swap(pth[parent], pth[child_1]);
			}

			break;
		}
		else
		{
			// Two children, need to compare to the smallest
			size_t min_child;

			if (get_time(pth[child_1]) < get_time(pth[child_2]))
				min_child = child_1;
			else
				min_child = child_2;


			if (get_time(pth[min_child]) < get_time(pth[parent]))
			{
				// Need to swap them in pht
				std::swap(pht[pth[parent]], pht[pth[min_child]]);

				// Need to swap parent and current in heap pth
				std::swap(pth[parent], pth[min_child]);
			}
			else
			{
				// time at parent is smaller than either child, heap property restored
				break;
			}

			parent = min_child;
		}

	}
}

void Sim::up_heapify(size_t disc_ind)
{
	size_t parent, child{ pht[disc_ind] };

	while (true)
	{
		// Determine index of parent node
		if (child % 2 == 0)
			parent = child / 2 - 1;
		else
			parent = child / 2;

		if (child == 0)
			break;

		if (get_time(pth[child]) < get_time(pth[parent]))
		{
			// Need to swap them in pht
			std::swap(pht[pth[parent]], pht[pth[child]]);

			// Need to swap parent and current in heap pth
			std::swap(pth[parent], pth[child]);

			child = parent;
		}
		else
		{
			// It is in the correct position in the heap
			break;
		}

	}

}

void Sim::update_time(size_t disc_ind, double new_t)
{
	double old_t{ get_time(disc_ind) };

	events_vec[disc_ind][new_vec[disc_ind]].t = new_t;

	if (new_t == old_t)
		return;
	
	if (new_t > old_t)
	{
		// Heap property is maintained automatically for parents of disc_ind's
		// current position. Can only be violated for disc_ind's children
		down_heapify(disc_ind);
	}
	else
	{
		// Heap property can only be violated for disc_ind's parents
		up_heapify(disc_ind);
	}
}

std::pair<size_t, double> Sim::get_next_wall_coll(size_t disc_ind)
{
	double current_best_t{ infinity };
	double current_wall_t;
	size_t best_wall{ size_t_max };

	// Ignore checking for sectors not adjacent to a wall
	size_t sector_ID{ initial_state[disc_ind].sector_ID };
	size_t x{ sector_ID % N }, y{ sector_ID / N };

	if (
		2 <= x && x <= N-3 &&
		2 <= y && y <= M-3
		)
		return { size_t_max, infinity };

	// which walls we need to check
	bool check_wall[] = {
		x == 1,			// Left
		y == M - 2,		// Right
		x == N - 2,		// Top
		y == 1			// Bottom
	};

	for (size_t wall_ind = 0; wall_ind < walls.size(); ++wall_ind)
	{
		if ((walls.size() == 4 && check_wall[wall_ind]) || walls.size() > 4)
		{
			current_wall_t = test_disc_wall_col(initial_state[disc_ind], events_vec[disc_ind][old_vec[disc_ind]], walls[wall_ind]);

			if (current_wall_t < current_best_t)
			{
				current_best_t = current_wall_t;
				best_wall = wall_ind;
			}
		}
	}

	return { best_wall, current_best_t };
}

std::pair<size_t, double> Sim::get_next_boundary_coll(size_t disc_ind)
{
	size_t sector_ID{ initial_state[disc_ind].sector_ID };

	// coordinates of sector
	size_t x{ sector_ID % N }, y{ sector_ID / N };

	// First N+1 boundaries correspond to vertical boundaries going from left to right
	// next M+1 are horizontal boundaries going from bottom to top
	size_t boundary_indices[] = {
		x,				// left
		x + 1,			// right
		N + 1 + y,		// bottom
		N + 1 + y + 1	// top
	};


	double current_best_t{ infinity };
	double current_boundary_t;
	size_t best_boundary{ size_t_max };

	const Event& e1{ events_vec[disc_ind][old_vec[disc_ind]] };

	for (size_t boundary_ind : boundary_indices)
	{
		// Ensure if the previous event was a boundary collision, we ignore the previously interacting boundary
		if (!(e1.disc_wall_col == Collision_Type::Disc_Boundary && e1.second_ind == boundary_ind))
		{
			Wall& boundary{ boundaries[boundary_ind] };

			current_boundary_t = test_disc_boundary_col(events_vec[disc_ind][old_vec[disc_ind]], boundary);

			if (current_boundary_t < current_best_t)
			{
				// Check we haven't already processed the collision and are currently on the boundary
				// Do this for cases where a disc crosses two boundaries at the same time
				Vec2D sector_centre{ bottom_left + Vec2D{ (x - 0.5) * sector_width, (y - 0.5) * sector_height } };

				Vec2D diff{ sector_centre - boundary.start };

				// normal vector to wall, points "towards" centre of sector disc is currently in
				Vec2D n{ diff - diff.dot(boundary.tangent) * boundary.tangent };

				// disc is entering current sector or travelling along its boundary, ignore boundary
				if (e1.new_v.dot(n) >= 0)
					continue;
				else
				{
					current_best_t = current_boundary_t;
					best_boundary = boundary_ind;
				}
			}
		}
	}

	return { best_boundary, current_best_t };
}

std::pair<size_t, double> Sim::get_next_disc_coll(size_t disc_ind)
{
	double best_time{ infinity };
	double p_ij;
	size_t best_ind{ size_t_max };

	const Disc& d1{ initial_state[disc_ind] };
	const Event& e1{ events_vec[disc_ind][old_vec[disc_ind]] };

	// sectors we need to check, starting with bottom left
	size_t sector_ID_arr[] = {
		d1.sector_ID - N - 1,
		d1.sector_ID - N,
		d1.sector_ID - N + 1,
		d1.sector_ID - 1,
		d1.sector_ID,
		d1.sector_ID + 1,
		d1.sector_ID + N - 1,
		d1.sector_ID + N,
		d1.sector_ID + N + 1,
	};

	for (size_t s_ID : sector_ID_arr)
	{
		for (size_t partner_ind : sector_entires[s_ID])
		{
			if (partner_ind != disc_ind)
			{
				const Disc& d2{ initial_state[partner_ind] };
				const Event& e2{ events_vec[partner_ind][old_vec[partner_ind]] };

				p_ij = test_disc_disc_col(d1, d2, e1, e2);

				// Ignore partner_ind if it isn't earlier than the next scheduled time
				// for partner_ind, ignore if its the same partner as e1.second_ind AND
				// at the same time as e1.t - avoids processing the same collision twice
				if (get_time(partner_ind) >= p_ij && p_ij < best_time &&
					!(e1.get_disc_partner() == partner_ind && p_ij == e1.t))
				{
					best_time = p_ij;
					best_ind = partner_ind;
				}
			}
		}
	}

	return {best_ind, best_time};
}

double Sim::solve_quadratic(const Vec2D & alpha, const Vec2D & beta, const double R)
{
	double a2{ alpha.mag2() };

	// When solving the quadratic, denominator is equal to zero
	if (a2 == 0.0)
		return std::numeric_limits<double>::infinity();

	double a_dot_b{ alpha.dot(beta) };
	double b2{ beta.mag2() };

	// coefficients in the quadratic, use algorithm from numerical recipes
	double a{a2};
	double b{a_dot_b};  // Note in derived formula there is a factor of 2, here it's cancelled out with the 2 in denominator of q
	double c{b2 - R*R};

	double disc;

	// Discriminant looks slighly odd as factor fo 4 is cancelled with the 2 in denominator of q
	disc = b*b - a*c;

	// Discriminant of quardatic is less than zero, no collision
	if (disc < 0.0)
		return std::numeric_limits<double>::infinity();

	double q;

	// No division by two as cancelled with factors in b & discriminant
	q = - (b + std::signbit(b)*std::sqrt(disc));

	double x1, x2;

	// Note a can't be zero as we have already checked a2 != 0.0
	x1 = q / a;
	x2 = c != 0.0 ? c/q : infinity;

	return std::min(x1, x2);
}

double Sim::test_disc_disc_col(const Disc & d1, const Disc & d2, const Event &e1, const Event &e2)
{
	Vec2D d1_pos{ e1.pos }, d2_pos{ e2.pos };
	double start_time;  // most recent time of e1.t or e2.t

	if (e1.t > e2.t)
	{
		start_time = e1.t;
		d2_pos += (e1.t - e2.t) * e2.new_v;
	}
	else
	{
		start_time = e2.t;
		d1_pos += (e2.t - e1.t) * e1.new_v;
	}

	Vec2D alpha{ e1.new_v - e2.new_v };
	Vec2D beta{ d1_pos - d2_pos };
	double R{ d1.R + d2.R };
	double t;

	t = Sim::solve_quadratic(alpha, beta, R);

	// If it is very small and negative, consider it to be exactly 0.0
	if (-5e-14 <= t && t < 0.0)
		t = 0.0;

	return t >= 0.0 ? t + start_time : std::numeric_limits<double>::infinity();
}

double Sim::test_disc_wall_col(const Disc& d, const Event & e, const Wall & w)
{	
	Vec2D delta{ e.pos - w.start };
	Vec2D alpha{ e.new_v - (e.new_v.dot(w.tangent)) * w.tangent };
	Vec2D beta{ delta - (delta.dot(w.tangent)) * w.tangent };
	
	Vec2D diff{ w.end - w.start };
	double s, t;

	t = Sim::solve_quadratic(alpha, beta, d.R);

	// If t is only slightly negative, it's likely a collision is meant to be
	// occuring at precisely t=0.0, so set it equal to zero
	if (t < 0.0 && t >= -5e-14)
		t = 0.0;

	s = (e.pos + e.new_v*t - w.start).dot(diff) / diff.mag2();

	if (t < 0.0 || s < 0.0 || s > 1.0)
		return infinity;

	return e.t + t;
}

double Sim::test_disc_boundary_col(const Event& e, const Wall& w)
{
	// All boundaries are either horizontal or vertical
	int coord_ind{ w.tangent[0] == 0.0 ? 0 : 1 };

	if (e.new_v[coord_ind] == 0.0)
		return infinity;

	double dt;

	dt = (w.start[coord_ind] - e.pos[coord_ind]) / e.new_v[coord_ind];

	// May need to be careful when ds is nearly zero
	return dt >= 0.0 ? e.t + dt : infinity;
}

void Sim::disc_wall_col(Event &e,  const Wall & w)
{
	Vec2D dv{ e.new_v - e.new_v.dot(w.tangent)*w.tangent };

	e.new_v -= 2 * dv;
}

void Sim::disc_disc_col(Disc & d1, Disc & d2, Event& e1, Event &e2)
{
	Vec2D du{ e2.new_v - e1.new_v };
	Vec2D dr{ e2.pos - e1.pos };
	double dr_mag{ d1.R + d2.R };

	double coeff{ 2 * du.dot(dr) / ((d1.m + d2.m)*dr_mag * dr_mag) };

	Vec2D dv{ coeff * dr };

	e1.new_v += d2.m*dv;
	e2.new_v -= d1.m*dv;
}

void Sim::advance(const Event &old_e, Event &new_e, double t)
{
	new_e.pos = old_e.pos + (t - old_e.t)*old_e.new_v;
	new_e.new_v = old_e.new_v;
}

size_t Sim::compute_sector_ID(const Vec2D& pos)
{
	double left, bottom;

	left = bottom_left[0];
	bottom = bottom_left[1];

	size_t x_ind, y_ind;

	x_ind = static_cast<size_t>((pos[0] - left) / sector_width) + 1;
	y_ind = static_cast<size_t>((pos[1] - bottom) / sector_height) + 1;

	return y_ind*N + x_ind;
}

void Sim::update_sector_ID(const size_t disc_ind, const Event& old_event)
{
	Disc& d{ initial_state[disc_ind] };

	// Only need to update sector IDs if we're dealing with a boundary collision
	if (old_event.disc_wall_col != Collision_Type::Disc_Boundary)
		return;

	// remove the current entry
	for (auto it = sector_entires[d.sector_ID].begin(); it != sector_entires[d.sector_ID].end(); ++it)
	{
		if (*it == disc_ind)
		{
			sector_entires[d.sector_ID].erase(it);
			break;
		}
	}

	// If we've just performed a boundary collision, we'll currently be on the boundary and
	// compute_sector_ID() might not give the right answer
	// coordinates of sector
	size_t sector_ID{ d.sector_ID };
	size_t x{ sector_ID % N }, y{ sector_ID / N };

	size_t boundary_ind{ old_event.second_ind };

	if (boundary_ind == x)  // left boundary
	{
		d.sector_ID = sector_ID - 1;
	}
	else if (boundary_ind == x + 1)  // right boundary
	{
		d.sector_ID = sector_ID + 1;
	}
	else if (boundary_ind == N + 1 + y)  // bottom boundary
	{
		d.sector_ID = sector_ID - N;
	}
	else  // top boundary
	{
		d.sector_ID = sector_ID + N;
	}
	
	sector_entires[d.sector_ID].push_back(disc_ind);
}

