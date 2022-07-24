// billiards: Program to model collisions between 2d discs
// Copyright (C) 2022  Tom Spencer (tspencerprog@gmail.com)
//
// This file is part of billiards
//
// billiards is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "Vec2D.h"

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

Vec2D operator+(const Vec2D& v)
{
	return v;
}

Vec2D operator-(const Vec2D& v)
{
	return Vec2D{ -v[0], v[1] };
}

std::ostream & operator<<(std::ostream & os, const Vec2D & v)
{
	os << v[0] << '\t' << v[1];

	return os;
}