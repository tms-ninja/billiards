#include "Sim.h"

void Sim::advance(size_t max_iterations, double max_t, bool record_events)
{
	// Implements the algorithm described in Lubachevsky 1991

	size_t current_it{ 0 };
	double current_t{ 0.0 };

	// true means we are still finding disc with their new event time of 0.0
	// Collisions for discs may not be performed in the order they take place
	// during this time, so we shouldn't count these as interactions as they may
	// be undone
	bool initial_setup{ true }; 

	size_t i, j, k, m;
	double P, Q, R;

	// Run to max_iterations*2 as disc-disc collisions count involve two iterations
	// Similarly, need to make disc-wall collisions count as 2
	while (current_it < 2*max_iterations && current_t < max_t)
	{
		//std::cout << current_it << '\n';

		//verify_heap();

		// Determine i
		i = get_min_time_ind();
		current_t = get_time(i);

		std::cout << "current_t = " << current_t << '\n';

		if (current_t > 0.0)
			initial_setup = false;

		// Swap new and old for disc i
		new_vec[i] = old_vec[i];
		old_vec[i] = 1 - new_vec[i];

		// Update count of how many collisions have been performed
		if (!initial_setup)
			if (events_vec[i][new_vec[i]].disc_wall_col)
				current_it += 2;
			else
				current_it += 1;

		// Save old state as that is the event that has just been processed
		if (!initial_setup && record_events)
			events.push_back(events_vec[i][old_vec[i]]);

		// Minimise over other discs
		std::tie(j, P) = get_next_disc_coll(i);

		// Minimise over boundaries
		std::tie(k, Q) = get_next_wall_coll(i);

		R = P < Q ? P : Q;
		//events_vec[i][new_vec[i]].t = R;

		update_time(i, R);

		// Set about advancing discs and performing collisions
		if (R < infinity)
		{
			Event& state_1{ events_vec[i][new_vec[i]] };

			Sim::advance(events_vec[i][old_vec[i]], state_1, R);

			// Disc-wall collision
			if (Q < P)
			{
				Sim::disc_wall_col(state_1, walls[k]);

				state_1.second_ind = k;
				state_1.disc_wall_col = true;

				//if (!initial_setup)
				//	current_it += 2;
			}
			else  // Disc-disc collision
			{
				Event& state_2{ events_vec[j][new_vec[j]] };

				//state_2.t = R;

				update_time(j, R);

				Sim::advance(events_vec[j][old_vec[j]], state_2, R);

				Sim::disc_disc_col(initial_state[i], initial_state[j], state_1, state_2);

				m = state_2.get_disc_partner();

				state_1.second_ind = j;
				state_2.second_ind = i;

				state_1.disc_wall_col = false;
				state_2.disc_wall_col = false;

				if (m != size_t_max && m != i)
				{
					Event& state_m{ events_vec[m][new_vec[m]] };

					Sim::advance(events_vec[m][old_vec[m]], state_m, state_m.t);

					state_m.second_ind = size_t_max;
				}

				//if (!initial_setup)
				//	current_it += 1;
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

		elem[0].t = 0.0;
		elem[0].pos = initial_state[ind].r;
		elem[0].new_v = initial_state[ind].v;
		elem[0].ind = ind;
		elem[0].second_ind = std::numeric_limits<size_t>::max();
		
		elem[1] = elem[0];

		// Add to the heap
		add_time_to_heap(ind);
	}
}

double Sim::get_time(size_t disc_ind)
{
	return events_vec[disc_ind][new_vec[disc_ind]].t;
}

size_t Sim::get_min_time_ind()
{
	return pth[0];
}

void Sim::add_time_to_heap(size_t disc_ind)
{
	pth.push_back(disc_ind);
	
	// Assume pht is of correct size
	pht[disc_ind] = pth.size() - 1;

	size_t parent, current{ pth.size() - 1 };

	while (true)
	{
		// Determine index of parent node
		if (current % 2 == 0)
			parent = current / 2 - 1;
		else
			parent = current / 2;

		if (current == 0)
			break;

		if (get_time(pth[current]) < get_time(pth[parent]))
		{
			// Need to swap them in pht
			std::swap(pht[pth[parent]], pht[pth[current]]);

			// Need to swap parent and current in heap pth
			std::swap(pth[parent], pth[current]);

			current = parent;
		}
		else
		{
			// It is in the correct position in the heap
			break;
		}

	}
}

void Sim::down_heapify(size_t disc_ind)
{
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
		// Heap property is maintained automatically for elements above disc_ind's
		// current position. Can only be violated for elements below disc_ind's
		// currnt position.
		down_heapify(disc_ind);
	}
	else
	{
		// Heap property can only be violated for elements above disc_ind's
		// current position
		up_heapify(disc_ind);
	}

	verify_heap();
}

void Sim::verify_heap()
{
	size_t child_1, child_2;
	bool error{ false };

	//std::cout << "Heap length: " << pth.size() << '\n';

	for (size_t parent = 0; parent < pth.size(); ++parent)
	{
		child_1 = 2 * parent + 1;
		child_2 = 2 * parent + 2;

		if (child_1 < pth.size() && get_time(pth[child_1]) < get_time(pth[parent]))
		{
			std::cout << "Invalid heap child 1: " << child_1 << "!\n";

			error = true;

			break;
		}

		if (child_2 < pth.size() && get_time(pth[child_2]) < get_time(pth[parent]))
		{
			std::cout << "Invalid heap child 2: " << child_2 << "!\n";

			error = true;

			break;
		}
			
	}

	if (error)
	{
		std::cout << "i\tpth[i]\tt\n";

		for (size_t i = 0; i < pth.size(); i++)
		{
			std::cout << i << '\t' << pth[i] << '\t' << get_time(pth[i]) << '\n';
		}
	}
}

std::pair<size_t, double> Sim::get_next_wall_coll(size_t disc_ind)
{
	double current_best_t{ infinity };
	double current_wall_t;
	size_t best_wall{ size_t_max };

	for (size_t wall_ind = 0; wall_ind < walls.size(); ++wall_ind)
	{
		current_wall_t = test_disc_wall_col(initial_state[disc_ind], events_vec[disc_ind][old_vec[disc_ind]], walls[wall_ind]);

		if (current_wall_t < current_best_t)
		{
			current_best_t = current_wall_t;
			best_wall = wall_ind;
		}
	}

	return { best_wall, current_best_t };
}

std::pair<size_t, double> Sim::get_next_disc_coll(size_t disc_ind)
{
	double best_time{ infinity };
	double p_ij;
	size_t best_ind{ size_t_max };

	const Disc& d1{ initial_state[disc_ind] };
	const Event& e1{ events_vec[disc_ind][old_vec[disc_ind]] };

	for (size_t partner_ind = 0; partner_ind < initial_state.size(); ++partner_ind)
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
	double b{2.0*a_dot_b};
	double c{b2 - R*R};

	double disc;

	disc = b*b - 4.0*a*c;

	// Discriminant of quardatic is less than zero, no collision
	if (disc < 0.0)
		return std::numeric_limits<double>::infinity();

	double q;

	q = - (b + std::signbit(b)*sqrt(disc))/ 2.0;

	double x1, x2;

	x1 = a != 0.0 ? q/a : infinity;
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

Vec2D Sim::disc_pos(const Disc & d, const double t)
{
	return d.r + t*d.v;
}

void Sim::advance(const Event &old_e, Event &new_e, double t)
{
	new_e.pos = old_e.pos + (t - old_e.t)*old_e.new_v;
	new_e.new_v = old_e.new_v;
}

