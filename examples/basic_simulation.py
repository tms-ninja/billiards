# billiards: Program to model collisions between 2d discs
# Copyright (C) 2022  Tom Spencer (tspencerprog@gmail.com)
#
# This file is part of billiards
#
# billiards is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Basic example of how to set up a simulation
import numpy as np

import billiards as bl

print("started")

# First setup the simulation
discs = [
    {'start': [3.0, 0.0], 'v': [1.0, 0.0], 'm': 1.0, 'R': 1.0},
    {'start': [7.0, np.sqrt(2.0)], 'v': [-1.0, 0.0], 'm': 4.0, 'R': 2.0},
]

s = bl.PySim([-10.0, -10.0], [10.0, 10.0])

for d in discs:
    s.add_disc(np.array(d['start']), np.array(d['v']), d['m'], d['R'])

# Perform one collision
s.advance(10, 10000.0, True)

# Print the events that occurred
for e in s.events:
    print(f"{e.t}\t{e.ind}\t{e.partner_ind}\t{e.col_type}\t{e.r}\t{e.v}")

print("End of program")

