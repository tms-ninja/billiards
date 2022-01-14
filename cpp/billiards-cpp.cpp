// billiards-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>

#include "billiards/Sim.h"

// For testing performance improvements are actually improvements
Sim setup_perf_test()
{
	Sim s{ {0.0, 0.0}, {10.0, 10.0} };

	//s.walls.push_back(Wall{ { 0.0,    0.0 }, {  0.0,  10.0 } });
	//s.walls.push_back(Wall{ { 0.0,   10.0 }, { 10.0,  10.0 } });
	//s.walls.push_back(Wall{ { 10.0,  10.0 }, { 10.0,   0.0 } });
	//s.walls.push_back(Wall{ { 10.0,   0.0 }, { 0.0,    0.0 } });

	Vec2D pos, v;
	double m{ 1.0 }, R{ 0.4 };
	double L{ 10.0 };  // width of box

	// Add balls on a 10 by 10 grid
	for (size_t i = 0; i < 10; ++i)
	{
		for (size_t j = 0; j < 10; ++j)
		{
			pos = { i + 0.5, j + 0.5 };

			v = { static_cast<double>((i * 17 + j) % 7), static_cast<double>((j * 13 + i) % 11) };
			v /= v.mag();

			s.initial_state.emplace_back(pos, v, m, R);
		}
	}

	s.setup();

	return s;
}

// Deliberately take it by copy
void run_perf_test(Sim s)
{
	s.advance(40000, 10000.0, true);

	std::cout << "Processed " << s.events.size() << " events, last event occured at " << s.events[s.events.size() - 1].t << '\n';
}

int main()
{
	Sim s{ setup_perf_test() };

	auto begin = std::chrono::steady_clock::now();

	for (size_t i = 0; i < 10; i++)
	{
		run_perf_test(s);
	}

	auto end = std::chrono::steady_clock::now();

	std::cout << "Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << "[ms]" << std::endl;

	//Sim s;

	//s.walls.push_back(Wall{ { 0.0,    0.0 }, {  0.0,  10.0 } });
	//s.walls.push_back(Wall{ { 0.0,   10.0 }, { 10.0,  10.0 } });
	//s.walls.push_back(Wall{ { 10.0,  10.0 }, { 10.0,   0.0 } });
	//s.walls.push_back(Wall{ { 10.0,   0.0 }, { 0.0,    0.0 } });

	//
	//s.initial_state.push_back(Disc{ {3.0, 5.0}, {  1.0, 0.0}, 1.0, 1.0 });
	//s.initial_state.push_back(Disc{ {7.0, 5.0 }, { -1.0, 0.0}, 1.0, 1.0 });
	//s.initial_state.push_back(Disc{ {8.5, 8.0 }, { 0.0, 0.0}, 1.0, 1.0 });
	//s.initial_state.push_back(Disc{ {5.0, 8.0 }, { 1.0, 0.0}, 1.0, 1.0 });

	//s.initial_state.push_back(Disc{ {2.0, 2.0 }, { -1.0, 0.0}, 1.0, 1.0 });
	//s.initial_state.push_back(Disc{ {6.0, 2.0 }, { 1.0, 0.0}, 1.0, 1.0 });

	//s.setup();

	//s.advance(20, 100.0, true);

	//for (Event& e : s.events)
	//	std::cout << e << '\n';
}
