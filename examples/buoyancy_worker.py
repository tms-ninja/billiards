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


from dataclasses import dataclass
import numpy as np
import billiards as bl

@dataclass
class SimProperties:
    """Data class to describe simulation properties"""
    bottom_left: np.ndarray
    top_right: np.ndarray
    initial_state: np.ndarray
    current_state: np.ndarray
    current_time: float

    def get_corrected_current_state(self):
        """Corrects current state postion so all particles position are at the same time"""
        cur_state = self.current_state.copy()
        cur_t = cur_state['t']
        cur_pos = cur_state['r']
        cur_v = cur_state['v']
        cur_pos += cur_v*(self.current_time - cur_t)[:, np.newaxis]
    
        return cur_state

def test_buoyancy(rho_central):
    """
    Runs a simulation for a central disc with the given density. Neutral 
    buoyancy corresponds to ~0.22

    Returns information about the simulation's properties and arrays containing 
    the position/times of the large central disc
    """
    
    # Setup the simulation
    L = 21.0*3  # Simulation width
    bottom_left = np.array([-L/2, -L/2])
    top_right = np.array([L/2, L/2])
    
    sim = bl.PySim(bottom_left, top_right, 3, 3)
    sim.g = np.array([0.0, -0.05])
    
    # Add big central disc
    R_central = 9.0
    m_central = rho_central*np.pi*R_central**2
    
    pos_central = np.zeros(2)
    v_central = np.zeros(2)
    
    sim.add_disc(
        r=pos_central,
        v=v_central,
        m=m_central,
        R=R_central
    )

    # Now add discs surrounding it
    R_small = 0.8
    m_small = 1.0
    
    # tolerance so everything is inside the simulation
    R_small_tol = R_small*1.01
    
    x_pos = np.linspace(bottom_left[0]+R_small_tol, top_right[0]-R_small_tol, 30)
    y_pos = np.linspace(bottom_left[1]+R_small_tol, top_right[1]-R_small_tol, 30)
    xx, yy = np.meshgrid(x_pos, y_pos)
    
    pos_small = np.stack((xx, yy)).reshape(2, -1).T
    
    # Only include discs that don't intersect with the big disc
    msk = np.linalg.norm(pos_small-pos_central, axis=1) > R_central + R_small
    pos_small = pos_small[msk]
    N_small = pos_small.shape[0]
    
    v_small = np.random.rand(N_small, 2)-0.5
    v_small /= np.linalg.norm(v_small, axis=1)[:, np.newaxis]
    
    # Finally add the discs
    for pos, v_s in zip(pos_small, v_small):
        sim.add_disc(pos, v_s, m_small, R_small)

    # Run the simulation for a bit
    sim.advance(200_000, 150.0, True)

    # Return initial state, true final state, central disc pos & t
    sim_prop = SimProperties(
        bottom_left=bottom_left,
        top_right=top_right,
        initial_state=sim.initial_state,
        current_state=sim.current_state,
        current_time=sim.current_time
    )

    # Now let's extract the x and y position of the big disc over time
    central_disc_events = [e for e in sim.events if e.ind==0]
    pos = np.array([pos_central] + [e.r for e in central_disc_events])
    t = np.array([0.0] + [e.t for e in central_disc_events])
    
    # Finally return the answer
    return sim_prop, t, pos