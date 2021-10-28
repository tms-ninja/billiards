# Basic example of how to set up a simulation
import numpy as np

import billiards as bl

# First setup the simulation
walls = [
    {'start': [ 0.0,  -10.0 ], 'end': [  0.0,  10.0 ]},
    {'start': [ 0.0,   10.0 ], 'end': [ 10.0,  10.0 ]},
    {'start': [ 10.0,  10.0 ], 'end': [ 10.0, -10.0 ]},
    {'start': [ 10.0, -10.0 ], 'end': [ 0.0,  -10.0 ]},
]

discs = [
    {'start': [3.0, 0.0], 'v': [1.0, 0.0], 'm': 1.0, 'R': 1.0},
    {'start': [7.0, np.sqrt(2.0)], 'v': [-1.0, 0.0], 'm': 4.0, 'R': 2.0},
]

s = bl.PySim()

for w in walls:
    s.add_wall(np.array(w['start']), np.array(w['end']))

for d in discs:
    s.add_disc(np.array(d['start']), np.array(d['v']), d['m'], d['R'])


# Set up the simulation ready to be run
s.setup()

# Perform one collision
s.advance(10, 10000.0, True)

for e in s.events:
    print(f"{e.t}\t{e.ind}\t{e.second_ind}\t{e.disc_wall_col}\t{e.pos}\t{e.new_v}")

print("End of program")

