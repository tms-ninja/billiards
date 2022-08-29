# billiards
 Models a two dimensional gas of billiards using the algorithm described in [*How to Simulate Billiards and Similar Systems*](https://arxiv.org/abs/cond-mat/0503627). It also makes use of rectangular sectoring to improve efficiency further. The simulation allows modelling rotating discs using constant coefficients of normal/tangential restitution (non-energy conserving choices are untested and may behave strangely) and discs experiencing a uniform gravitational field in an arbritrary direction. Discs with differing masses, radii and moments of inertia are also supported.

Requires Cython, NumPy and SciPy to be built. Other libraries (such as Matplotlib) may further be required for examples. The project can be built by running the `setup.py` using

`python setup.py build_ext --inplace`

It can then be imported as simply `import billiards`. See `help(billiards)` for documentation.
