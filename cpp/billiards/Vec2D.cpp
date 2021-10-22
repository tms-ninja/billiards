#include "Vec2D.h"

Vec2D::Vec2D()
{
}

Vec2D::Vec2D(double x, double y)
	: data{ x, y }
{
}

double & Vec2D::operator[](int ind)
{
	return data[ind];
}

const double & Vec2D::operator[](int ind) const
{
	return data[ind];
}
