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

# Worker module for equation-of-state.ipynb

from dataclasses import dataclass
import numpy as np
import billiards as bl


def compute_sim_pressure(sim, bottom_left, top_right):
    """Computes the pressure using the simulation provided"""

    cur_v = sim.initial_state['v']
    masses = sim.initial_state['m']
    total_dP = 0.0  # total change in momentum
    
    for e in sim.events:
        if e.col_type==bl.PyCol_Type.Disc_Wall:
            
            dv = e.v - cur_v[e.ind]
            
            total_dP += masses[e.ind] * np.linalg.norm(dv)
        
        cur_v[e.ind] = e.v

    # Pressure if total change in momentum / time over which momentum change occured / "area" of box's surface
    dt = sim.events[-1].t - sim.events[0].t
    A = 2*np.sum(top_right - bottom_left)  # As we're in 2d, the "area" is really perimeter of box
    
    return total_dP / (dt * A)

def measure_pressure(n, kB_T, R):
    """
    Determines the pressure for a gas with number density n and temperature T by running a simulation
    """
    
    # Number, speed, masses, radii of discs
    N_discs = 2_500
    v = 1.0
    m = 1.0
    
    # Setup the simulation
    L = np.sqrt(N_discs / n)  # Simulation width
    bottom_left = np.array([-L/2, -L/2])
    top_right = np.array([L/2, L/2])

    sim = bl.PySim(bottom_left, top_right, 49, 49)

    sim.add_random_discs(bottom_left, top_right, N_discs, m, R, kB_T=kB_T, pos_allocation='grid')
    
    sim.advance(100_000, 10_000.0, True)
    
    return compute_sim_pressure(sim, bottom_left, top_right)

