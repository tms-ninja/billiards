// billiards-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "billiards/Sim.h"

int main()
{
	Sim s;

	s.walls.push_back(Wall{ { 0.0,  -10.0 }, {  0.0,  10.0 } });
	s.walls.push_back(Wall{ { 0.0,   10.0 }, { 10.0,  10.0 } });
	s.walls.push_back(Wall{ { 10.0,  10.0 }, { 10.0, -10.0 } });
	s.walls.push_back(Wall{ { 10.0, -10.0 }, { 0.0,  -10.0 } });

	s.current_state.push_back(Disc{ {3.0, 0.0}, {1.0, 0.0}, 1.0, 1.0 });
	s.current_state.push_back(Disc{ {7.0, sqrt(2)}, {-1.0, 0.0}, 1.0, 1.0 });

	s.advance(1, 100.0, true);

	for (Event& e : s.events)
		std::cout << e << '\n';
}
