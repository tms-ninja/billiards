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

Vec2D & Vec2D::operator+=(const Vec2D & v)
{
	data[0] += v[0];
	data[1] += v[1];

	return *this;
}

Vec2D & Vec2D::operator-=(const Vec2D & v)
{
	data[0] -= v[0];
	data[1] -= v[1];

	return *this;
}

Vec2D & Vec2D::operator*=(const Vec2D & v)
{
	data[0] *= v[0];
	data[1] *= v[1];

	return *this;
}

Vec2D & Vec2D::operator/=(const Vec2D & v)
{
	data[0] /= v[0];
	data[1] /= v[1];

	return *this;
}

Vec2D & Vec2D::operator+=(const double v)
{
	data[0] += v;
	data[1] += v;

	return *this;
}

Vec2D & Vec2D::operator-=(const double v)
{
	data[0] -= v;
	data[1] -= v;

	return *this;
}

Vec2D & Vec2D::operator*=(const double v)
{
	data[0] *= v;
	data[1] *= v;

	return *this;
}

Vec2D & Vec2D::operator/=(const double v)
{
	data[0] /= v;
	data[1] /= v;

	return *this;
}

Vec2D operator+(Vec2D v1, const Vec2D & v2)
{
	v1 += v2;

	return v1;
}

Vec2D operator-(Vec2D v1, const Vec2D & v2)
{
	v1 -= v2;

	return v1;
}

Vec2D operator*(Vec2D v1, const Vec2D & v2)
{
	v1 *= v2;

	return v1;
}

Vec2D operator/(Vec2D v1, const Vec2D & v2)
{
	v1 /= v2;

	return v1;
}

Vec2D operator+(Vec2D v1, const double n)
{
	v1 += n;

	return v1;
}

Vec2D operator-(Vec2D v1, const double n)
{
	v1 -= n;

	return v1;
}

Vec2D operator*(Vec2D v1, const double n)
{
	v1 *= n;

	return v1;
}

Vec2D operator/(Vec2D v1, const double n)
{
	v1 /= n;

	return v1;
}

Vec2D operator+(const double n, Vec2D v1)
{
	v1 += n;

	return v1;
}

Vec2D operator-(const double n, Vec2D v1)
{
	v1[0] = n - v1[0];
	v1[1] = n - v1[1];
	
	return v1;
}

Vec2D operator*(const double n, Vec2D v1)
{
	v1 *= n;

	return v1;
}

Vec2D operator/(const double n, Vec2D v1)
{
	v1[0] = n / v1[0];
	v1[1] = n / v1[1];

	return v1;
}
