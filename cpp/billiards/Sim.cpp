#include "Sim.h"

void Sim::advance(size_t max_iterations, double max_t, bool record_events)
{
	size_t current_it{ 0 };
	double time_to_col, current_t{ 0.0 };

	Event current_best;

	current_best.t = std::numeric_limits<double>::infinity();

	while (current_it < max_iterations && current_t < max_t)
	{
		// Determine next collision
		for (size_t d_ind = 0; d_ind < current_state.size(); ++d_ind)
		{
			// Check walls
			for (size_t w_ind = 0; w_ind < walls.size(); ++w_ind)
			{
				time_to_col = Sim::test_disc_wall_col(current_state[d_ind], walls[w_ind]);

				if (time_to_col >= 0.0 && time_to_col < current_best.t)
				{
					current_best.disc_wall_col = true;
					current_best.t = time_to_col;
					current_best.ind = d_ind;
					current_best.second_ind = w_ind;
				}
			}

			// Check discs
			for (size_t second_ind = d_ind + 1; second_ind < current_state.size(); ++second_ind)
			{
				time_to_col = Sim::test_disc_disc_col(current_state[d_ind], current_state[second_ind]);

				if (time_to_col >= 0.0 && time_to_col < current_best.t)
				{
					current_best.disc_wall_col = false;
					current_best.t = time_to_col;
					current_best.ind = d_ind;
					current_best.second_ind = second_ind;
				}
			}
		}

		// Advance all the discs to the moment of collision
		for (Disc& d : current_state)
			d.pos += current_best.t * d.v;

		// Now we know which disc and disc/wall are next to be hit, perform the collision
		if (current_best.disc_wall_col)
			Sim::disc_wall_col(current_state[current_best.ind], walls[current_best.second_ind]);
		else
			Sim::disc_disc_col(current_state[current_best.ind], current_state[current_best.second_ind]);

		// Save the event is necessary
		if (record_events)
		{
			const Disc& d{ current_state[current_best.ind] };

			current_best.t += current_t;
			current_best.pos = d.pos;
			current_best.new_v = d.v;

			events.push_back(current_best);	

			if (!current_best.disc_wall_col)
			{
				const Disc& second_disc{ current_state[current_best.second_ind] };

				std::swap(current_best.ind, current_best.second_ind);

				current_best.pos = second_disc.pos;
				current_best.new_v = second_disc.v;

				events.push_back(current_best);
			}
		}

		current_it += 1;
		current_t = current_best.t;

		current_best.t = std::numeric_limits<double>::infinity();
	}
}

double Sim::solve_quadratic(const Vec2D & alpha, const Vec2D & beta, const double R)
{
	double a2{ alpha.mag2() };

	// When solving the quadratic, denominator is equal to zero
	if (a2 == 0.0)
		return std::numeric_limits<double>::infinity();

	double a_dot_b{ alpha.dot(beta) };
	double b2{ beta.mag2() };
	double disc;

	disc = a_dot_b * a_dot_b + a2 * (R * R - b2);

	// Discriminant of quardatic is less than zero, no collision
	if (disc < 0.0)
		return std::numeric_limits<double>::infinity();

	return -(a_dot_b + sqrt(disc)) / a2;
}

double Sim::test_disc_disc_col(const Disc & d1, const Disc & d2)
{
	Vec2D alpha{ d1.v - d2.v };
	Vec2D beta{ d1.pos - d2.pos };
	double R{ d1.R + d2.R };
	double t;

	t = Sim::solve_quadratic(alpha, beta, R);

	return t >= 0.0 ? t : std::numeric_limits<double>::infinity();
}

double Sim::test_disc_wall_col(const Disc & d, const Wall & w)
{	
	Vec2D delta{ d.pos - w.start };
	Vec2D alpha{ d.v - (d.v.dot(w.tangent)) * w.tangent };
	Vec2D beta{ delta - (delta.dot(w.tangent)) * w.tangent };
	
	Vec2D diff{ w.end - w.start };
	double s, t;

	t = Sim::solve_quadratic(alpha, beta, d.R);
	s = (d.r(t) - w.start).dot(diff) / diff.mag2();

	if (t < 0.0 || s < 0.0 || s > 1.0)
		return std::numeric_limits<double>::infinity();

	return t;
}

void Sim::disc_wall_col(Disc & d, const Wall & w)
{
	Vec2D dv{ d.v - d.v.dot(w.tangent)*w.tangent };

	d.v -= 2 * dv;
}

void Sim::disc_disc_col(Disc & d1, Disc & d2)
{
	Vec2D du{ d2.v - d1.v };
	Vec2D dr{ d2.pos - d1.pos };
	double dr_mag{ d1.R + d2.R };

	double coeff{ 2 * du.dot(dr) / ((d1.m + d2.m)*dr_mag * dr_mag) };

	Vec2D dv{ coeff * dr };

	d1.v += d2.m*dv;
	d2.v -= d1.m*dv;
}

