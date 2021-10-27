#pragma once
#include "Vec2D.h"

class Disc
{
public:
	Vec2D r;
	Vec2D v;
	double m;
	double R;

	Disc();

	Disc(const Vec2D &pos, const Vec2D &v, double m, double R);
};

