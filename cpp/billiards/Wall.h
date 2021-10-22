#pragma once
#include "Vec2D.h"

class Wall
{
public:
	Vec2D start;
	Vec2D end;
	Vec2D tangent;  // tangent unit vector in diection start -> end

	Wall(const Vec2D &start, const Vec2D &end);
};

