#pragma once
#include "Vec2D.h"

class Disc
{
public:
	Vec2D pos;
	Vec2D v;
	double m;
	double R;

	Disc(const Vec2D &pos, const Vec2D &v, double m, double R);
};

