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
