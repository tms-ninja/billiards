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

	// computes the position of the disc at time t
	Vec2D r(double t) const;
};

