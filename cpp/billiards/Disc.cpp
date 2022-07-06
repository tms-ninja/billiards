
#include "Disc.h"

Disc::Disc()
{
}

Disc::Disc(const Vec2D & pos, const Vec2D & v, double m, double R, size_t sector_ID)
	: r{ pos }, v{ v }, m{ m }, R{ R }, sector_ID{sector_ID}
{
}
