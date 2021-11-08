// billiards-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>

#include "billiards/Sim.h"

// For testing performance improvements are actually improvements
Sim setup_perf_test()
{
	Sim s;

	s.walls.push_back(Wall{ { 0.0,    0.0 }, {  0.0,  10.0 } });
	s.walls.push_back(Wall{ { 0.0,   10.0 }, { 10.0,  10.0 } });
	s.walls.push_back(Wall{ { 10.0,  10.0 }, { 10.0,   0.0 } });
	s.walls.push_back(Wall{ { 10.0,   0.0 }, { 0.0,    0.0 } });

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

			s.current_state.emplace_back(pos, v, m, R);
		}
	}

	return s;
}

// Deliberately take it by copy
void run_perf_test(Sim s)
{
	s.advance(40000, 10000.0, true);
}

int main()
{

	Sim s{ setup_perf_test() };

	auto begin = std::chrono::steady_clock::now();

	run_perf_test(s);

	auto end = std::chrono::steady_clock::now();

	std::cout << "Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << "[ms]" << std::endl;
}
