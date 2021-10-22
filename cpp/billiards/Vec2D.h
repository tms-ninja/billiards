#pragma once
#include <cmath>

class Vec2D
{
	double data[2];

public:
	Vec2D();

	Vec2D(double x, double y);

	double& operator[](int ind);

	const double& operator[](int ind) const;

	inline double mag() const
	{
		return hypot(data[0], data[1]);
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

