
#include "Disc.h"

Disc::Disc(Vec2D & pos, Vec2D & v, double m, double R)
	: pos{ pos }, v{ v }, m{ m }, R{ R }
{
}

Vec2D Disc::r(double t) const
{
	return pos + t * v;
}
