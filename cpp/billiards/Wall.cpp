#include "Wall.h"

Wall::Wall(const Vec2D & start, const Vec2D & end)
	: start{ start }, end{ end }
{
	tangent = end - start;
	tangent /= tangent.mag();
}
