#pragma once
#include "Vec2D.h"

class Disc
{
public:
	Vec2D pos;
	Vec2D v;
	double m;
	double R;

	Disc(Vec2D &pos, Vec2D &v, double m, double R);
};

