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

#include <iostream>
#include <chrono>

#include "billiards/Sim.h"

// For testing performance improvements are actually improvements
Sim setup_perf_test(bool gravity)
{
	Sim s{ {0.0, 0.0}, {10.0, 10.0}, 10, 10 };

	if (gravity)
		s.set_g({ 0.0, -1.0 });

	Vec2D pos, v;
	double m{ 1.0 }, R{ 0.4 };
	double I{ m*R*R / 2.0 }, w{ 0.0 };
	double L{ 10.0 };  // width of box

	// Add balls on a 10 by 10 grid
	for (size_t i = 0; i < 10; ++i)
	{
		for (size_t j = 0; j < 10; ++j)
		{
			pos = { i + 0.5, j + 0.5 };

			v = { static_cast<double>((i * 17 + j) % 7), static_cast<double>((j * 13 + i) % 11) };

			if (v.mag() == 0.0)
				v = { 1.0, 0.0 };

			v /= v.mag();

			s.add_disc(pos, v, w, m, R, I);
		}
	}

	return s;
}

// Deliberately take it by copy so we start with a fresh simulation each time
void run_perf_test(Sim s)
{
	s.advance(40000, 10000.0, true);

	std::cout << "Processed " << s.events.size() << " events, last event occured at " << s.events[s.events.size() - 1].t << '\n';
}

int main()
{
	std::cout << "Started\n";
	Sim s{ setup_perf_test(false) };

	auto begin = std::chrono::steady_clock::now();

	for (size_t i = 0; i < 40; i++)
	{
		run_perf_test(s);
	}

	auto end = std::chrono::steady_clock::now();

	std::cout << "Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << "[ms]" << std::endl;




	//Sim s{ { 0.0,    0.0 }, { 10.0,  10.0 }, 4, 4 };

	
	//s.add_disc( { 2.0, 4.0 }, {1.0, 0.0}, 1.0, 1.0 );
	//s.add_disc({ 7.0, 4.0 }, { -1.0, 0.0 }, 1.0, 1.0);
	//s.initial_state.push_back(Disc{ {7.0, 5.0 }, { -1.0, 0.0}, 1.0, 1.0 });
	//s.initial_state.push_back(Disc{ {8.5, 8.0 }, { 0.0, 0.0}, 1.0, 1.0 });
	//s.initial_state.push_back(Disc{ {5.0, 8.0 }, { 1.0, 0.0}, 1.0, 1.0 });

	//s.initial_state.push_back(Disc{ {2.0, 2.0 }, { -1.0, 0.0}, 1.0, 1.0 });
	//s.initial_state.push_back(Disc{ {6.0, 2.0 }, { 1.0, 0.0}, 1.0, 1.0 });

	//s.setup();

	//std::cout << "started\n";

	//s.advance(20, 100.0, true);

	//for (Event& e : s.events)
	//	std::cout << e << '\n';
}
