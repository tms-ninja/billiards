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

#pragma once
#include <cmath>
#include <iostream>

class Vec2D
{
	double data[2];

public:
	Vec2D() = default;

	Vec2D(double x, double y);

	double& operator[](int ind);
	const double& operator[](int ind) const;

	// Elementwise arithmetic operators
	Vec2D& operator+=(const Vec2D &v);
	Vec2D& operator-=(const Vec2D &v);
	Vec2D& operator*=(const Vec2D &v);
	Vec2D& operator/=(const Vec2D &v);

	Vec2D& operator+=(const double v);
	Vec2D& operator-=(const double v);
	Vec2D& operator*=(const double v);
	Vec2D& operator/=(const double v);



	inline double mag() const
	{
		return std::hypot(data[0], data[1]);
	}

	inline double mag2() const
	{
		return data[0] * data[0] + data[1] * data[1];
	}

	inline double dot(const Vec2D &v) const
	{
		return data[0] * v[0] + data[1] * v[1];
	}

};

// Elementwise binary artihmetic operators
Vec2D operator+(Vec2D v1, const Vec2D &v2);
Vec2D operator-(Vec2D v1, const Vec2D &v2);
Vec2D operator*(Vec2D v1, const Vec2D &v2);
Vec2D operator/(Vec2D v1, const Vec2D &v2);

Vec2D operator+(Vec2D v1, const double n);
Vec2D operator-(Vec2D v1, const double n);
Vec2D operator*(Vec2D v1, const double n);
Vec2D operator/(Vec2D v1, const double n);

Vec2D operator+(const double n, Vec2D v1);
Vec2D operator-(const double n, Vec2D v1);
Vec2D operator*(const double n, Vec2D v1);
Vec2D operator/(const double n, Vec2D v1);

Vec2D operator+(const Vec2D& v);
Vec2D operator-(const Vec2D& v);

std::ostream& operator<<(std::ostream& os, const Vec2D& v);
