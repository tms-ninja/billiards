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

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import billiards as bl

def total_KE(m, v):
    ke = m * np.linalg.norm(v, axis=1)**2

    return np.sum(ke) / 2.0

def create_20_disc_sim(n_sector_x, n_sector_y):
    """
    Creates and runs a simulation containing 20 discs for verification pruposes
    Can choose the number of sectors to be used in the simulation
    """

    # Set the seed so we know add_random_discs() won't fail
    np.random.seed(30)

    box_bottom_left = [ 0.0,  0.0]
    box_top_right = [20.0, 20.0]

    s = bl.PySim(box_bottom_left, box_top_right, n_sector_x, n_sector_y)

    #s.add_box_walls(box_bottom_left, box_top_right)
    s.add_random_discs(np.array(box_bottom_left), np.array(box_top_right), 20, 5.0, 1.0, 1.0)

    s.setup()

    # Perform some collision
    s.advance(1000, 100.0, True)

    return {'box': [box_bottom_left, box_top_right], 's': s}


class Test_PySim(unittest.TestCase):
    
    def test_disc_wall_col_vertical(self):
        """Tests discs collide with vertical walls correctly"""
        s = bl.PySim([0.0, 0.0], [10.0, 10.0])

        s.add_disc([3.0, 3.0], [-1.0, 1.0], 1.0, 1.0)

        s.setup()

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([1.0, 5.0])
        expected_v = np.array([1.0, 1.0])

        self.assertAlmostEqual(ev.t, 2.0)
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.second_ind, 0)
        self.assertEqual(ev.disc_wall_col, True)
        assert_allclose(ev.pos, expected_pos)
        assert_allclose(ev.new_v, expected_v)

    def test_disc_wall_col_horizontal(self):
        """Tests discs collide with horizontal walls correctly"""
        s = bl.PySim([0.0, 0.0], [10.0, 10.0])

        s.add_disc([3.0, 7.0], [1.0, 1.0], 1.0, 1.0)

        s.setup()

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([5.0, 9.0])
        expected_v = np.array([1.0, -1.0])

        self.assertAlmostEqual(ev.t, 2.0)
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.second_ind, 1)
        self.assertEqual(ev.disc_wall_col, True)
        assert_allclose(ev.pos, expected_pos)
        assert_allclose(ev.new_v, expected_v)

    def test_disc_wall_col_diagonal(self):
        """Tests discs collide with diagonal walls correctly"""
        s = bl.PySim([-10.0, -10.0], [10.0, 10.0])

        s.add_wall([0.0, 0.0], [10.0, 10.0])
        s.add_disc([3.0, 0.0], [0.0, 1.0], 1.0, 1.0)

        s.setup()

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([3.0, 3.0 - 1 / np.sin(np.pi/4)])
        expected_v = np.array([1.0, 0.0])

        self.assertAlmostEqual(ev.t, 3.0 - 1 / np.sin(np.pi/4))
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.second_ind, 4)
        self.assertEqual(ev.disc_wall_col, True)
        assert_allclose(ev.pos, expected_pos)
        assert_allclose(ev.new_v, expected_v, atol=1e-15)

    def test_disc_disc_col_vertical(self):
        """Tests vertial disc-disc collisions"""
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([3.0, 0.0], [0.0, 1.0], 1.0, 1.0)
        s.add_disc([3.0, 10.0], [0.0, -1.0], 1.0, 1.0)

        s.setup()
        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'second_ind': 1,
                'disc_wall_col': False,
                'pos': np.array([3.0, 4.0]),
                'new_v': np.array([0.0, -1.0]),
            },
            {
                't': 4.0,
                'ind': 1,
                'second_ind': 0,
                'disc_wall_col': False,
                'pos': np.array([3.0, 6.0]),
                'new_v': np.array([0.0, 1.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.second_ind, exp['second_ind'])
            self.assertEqual(ev.disc_wall_col, exp['disc_wall_col'])
            assert_allclose(ev.pos, exp['pos'])
            assert_allclose(ev.new_v, exp['new_v'])

    def test_disc_disc_col_horizontally(self):
        """Tests horizontal disc-dsic collisions"""
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 1.0)

        s.setup()
        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'second_ind': 1,
                'disc_wall_col': False,
                'pos': np.array([4.0, 0.0]),
                'new_v': np.array([-1.0, 0.0]),
            },
            {
                't': 4.0,
                'ind': 1,
                'second_ind': 0,
                'disc_wall_col': False,
                'pos': np.array([6.0, 0.0]),
                'new_v': np.array([1.0, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.second_ind, exp['second_ind'])
            self.assertEqual(ev.disc_wall_col, exp['disc_wall_col'])
            assert_allclose(ev.pos, exp['pos'])
            assert_allclose(ev.new_v, exp['new_v'])
    
    def test_disc_disc_col_off_centre(self):
        """
        Tests off centre disc-disc collisions, where disc velocity isn't
        towards the other disc's centre.
        """
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0)
        s.add_disc([10.0, np.sqrt(2)], [-1.0, 0.0], 1.0, 1.0)

        s.setup()
        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 5 - np.sqrt(2)/2,
                'ind': 0,
                'second_ind': 1,
                'disc_wall_col': False,
                'pos': np.array([5 - np.sqrt(2)/2, 0.0]),
                'new_v': np.array([0.0, -1.0]),
            },
            {
                't': 5 - np.sqrt(2)/2,
                'ind': 1,
                'second_ind': 0,
                'disc_wall_col': False,
                'pos': np.array([5 + np.sqrt(2)/2, np.sqrt(2)]),
                'new_v': np.array([0.0, 1.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.second_ind, exp['second_ind'])
            self.assertEqual(ev.disc_wall_col, exp['disc_wall_col'])
            assert_allclose(ev.pos, exp['pos'])
            assert_allclose(ev.new_v, exp['new_v'], atol=1e-14)

    def test_disc_disc_col_speed(self):
        """Tests disc-disc collisions where each disc has a different speed"""
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [2.0, 0.0], 1.0, 1.0)
        s.add_disc([10.0, 0.0], [-3.0, 0.0], 1.0, 1.0)

        s.setup()
        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 1.6,
                'ind': 0,
                'second_ind': 1,
                'disc_wall_col': False,
                'pos': np.array([3.2, 0.0]),
                'new_v': np.array([-3.0, 0.0]),
            },
            {
                't': 1.6,
                'ind': 1,
                'second_ind': 0,
                'disc_wall_col': False,
                'pos': np.array([5.2, 0.0]),
                'new_v': np.array([2.0, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.second_ind, exp['second_ind'])
            self.assertEqual(ev.disc_wall_col, exp['disc_wall_col'])
            assert_allclose(ev.pos, exp['pos'])
            assert_allclose(ev.new_v, exp['new_v'])
    
    def test_disc_disc_col_mass(self):
        """Tests disc-disc collisions where each disc has a different mass"""
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 2.0, 1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 3.0, 1.0)

        s.setup()
        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'second_ind': 1,
                'disc_wall_col': False,
                'pos': np.array([4.0, 0.0]),
                'new_v': np.array([-1.4, 0.0]),
            },
            {
                't': 4.0,
                'ind': 1,
                'second_ind': 0,
                'disc_wall_col': False,
                'pos': np.array([6.0, 0.0]),
                'new_v': np.array([0.6, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.second_ind, exp['second_ind'])
            self.assertEqual(ev.disc_wall_col, exp['disc_wall_col'])
            assert_allclose(ev.pos, exp['pos'])
            assert_allclose(ev.new_v, exp['new_v'])
    
    def test_disc_disc_col_radii(self):
        """Tests disc-disc collisions where each disc has a different radius"""
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 2.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 3.0)

        s.setup()
        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 2.5,
                'ind': 0,
                'second_ind': 1,
                'disc_wall_col': False,
                'pos': np.array([2.5, 0.0]),
                'new_v': np.array([-1.0, 0.0]),
            },
            {
                't': 2.5,
                'ind': 1,
                'second_ind': 0,
                'disc_wall_col': False,
                'pos': np.array([7.5, 0.0]),
                'new_v': np.array([1.0, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.second_ind, exp['second_ind'])
            self.assertEqual(ev.disc_wall_col, exp['disc_wall_col'])
            assert_allclose(ev.pos, exp['pos'])
            assert_allclose(ev.new_v, exp['new_v'])
    
    def test_KE_conservation(self):
        """
        Tests kinetic energy is approximately conserved during a 
        test simulation
        """
        
        sim = create_20_disc_sim(1, 1)

        s = sim['s']

        initial_state = s.initial_state

        initial_KE = total_KE(initial_state['m'], initial_state['v'])

        for c_state in s.replay_by_time(0.1):
            m, v = c_state['m'], c_state['v']

            current_KE = total_KE(m, v)

            self.assertAlmostEqual(current_KE, initial_KE)

    def test_out_of_bounds(self):
        """
        Tests discs don't go out of bounds during a test simulation
        """

        sim = create_20_disc_sim(1, 1)

        s = sim['s']
        box_bottom_left, box_top_right = sim['box']

        left, bottom = box_bottom_left
        right, top = box_top_right

        R = s.initial_state['R']

        for c_state in s.replay_by_time(0.1):
            pos = c_state['r']

            np.testing.assert_array_equal(left + R <= pos[:, 0], True)
            np.testing.assert_array_equal(pos[:, 0] <= right - R, True)

            np.testing.assert_array_equal(bottom + R <= pos[:, 1], True)
            np.testing.assert_array_equal(pos[:, 1] <= top - R, True)

    def test_overlapping(self):
        """
        Tests discs don't overlap during a test simulation, no sectoring used
        """

        sim = create_20_disc_sim(1, 1)  # No sectoring used
        s = sim['s']

        R = s.initial_state['R']

        for c_state in s.replay_by_time(0.1):
            pos = c_state['r']

            for i in range(pos.shape[0]):

                diff = pos[i+1:] - pos[i] 

                dist = np.linalg.norm(diff, axis=1)

                np.testing.assert_array_equal(R[i] + R[i+1:] <= dist, True)
    
    def test_overlapping_sectoring(self):
        """
        Tests discs don't overlap during a test simulation, sectoring used
        """

        sim = create_20_disc_sim(9, 9)  # sectoring used
        s = sim['s']

        R = s.initial_state['R']

        for c_state in s.replay_by_time(0.1):
            pos = c_state['r']

            for i in range(pos.shape[0]):

                diff = pos[i+1:] - pos[i] 

                dist = np.linalg.norm(diff, axis=1)

                np.testing.assert_array_equal(R[i] + R[i+1:] <= dist, True)

    def test_get_bounds(self):
        """Tests get_sim_bounds property"""

        bottom_left = np.array([1.0, 2.0])
        top_right = np.array([3.0, 4.0])

        s = bl.PySim(bottom_left, top_right)

        bounds = s.bounds

        assert_array_equal(bounds[0], bottom_left)
        assert_array_equal(bounds[1], top_right)

    def test_reject_adding_disc_outside_sim(self):
        """
        Tests add_disc() rejects attempting to add a disc outside the simulation
        Expected that it raises a ValueError
        """

        L = 10.0  # Width/height of simulation box

        bottom_left = [0.0, 0.0]
        top_right = [L, L]

        s = bl.PySim(bottom_left, top_right)

        v = [0.0, 0.0]
        m = 1.0
        R = 1.0

        # Test both a disc partially and fully outside the simulation
        # Left 
        with self.assertRaises(ValueError) as _:
            s.add_disc([R / 2, L / 2], v, m, R)
        
        with self.assertRaises(ValueError) as _:
            s.add_disc([-L, L / 2], v, m, R)

        # Right
        with self.assertRaises(ValueError) as _:
            s.add_disc([L - R / 2, L / 2], v, m, R)
        
        with self.assertRaises(ValueError) as _:
            s.add_disc([2 * L, L / 2], v, m, R)

        # Bottom
        with self.assertRaises(ValueError) as _:
            s.add_disc([L / 2, R / 2], v, m, R)
        
        with self.assertRaises(ValueError) as _:
            s.add_disc([L / 2, -L], v, m, R)

        # Top
        with self.assertRaises(ValueError) as _:
            s.add_disc([L / 2, L - R / 2], v, m, R)
        
        with self.assertRaises(ValueError) as _:
            s.add_disc([L / 2, 2 * L], v, m, R)

    def test_reject_adding_negative_mass_disc(self):
        """
        Tests add_disc() rejects attempting to add a disc with mass less than 
        or equal to 0
        Expected that it raises a ValueError
        """

        L = 10.0  # Width/height of simulation box

        bottom_left = [0.0, 0.0]
        top_right = [L, L]

        s = bl.PySim(bottom_left, top_right)

        pos = [L/2, L/2]
        v = [0.0, 0.0]
        m = 1.0
        R = 1.0

        # Ensure it can't equal zero
        with self.assertRaises(ValueError) as _:
            s.add_disc(pos, v, 0.0, R)

        # Ensure it can't be less than zero
        with self.assertRaises(ValueError) as _:
            s.add_disc(pos, v, -1.0, R)
    
    def test_reject_adding_negative_radius_disc(self):
        """
        Tests add_disc() rejects attempting to add a disc with radius less than 
        or equal to 0
        Expected that it raises a ValueError
        """

        L = 10.0  # Width/height of simulation box

        bottom_left = [0.0, 0.0]
        top_right = [L, L]

        s = bl.PySim(bottom_left, top_right)

        pos = [L/2, L/2]
        v = [0.0, 0.0]
        m = 1.0

        # Ensure it can't equal zero
        with self.assertRaises(ValueError) as _:
            s.add_disc(pos, v, m, 0.0)

        # Ensure it can't be less than zero
        with self.assertRaises(ValueError) as _:
            s.add_disc(pos, v, m, -1.0)

    def test_reject_adding_disc_larger_than_sector_size(self):
        """
        Tests add_disc() rejects attempting to add a disc with diameter larger 
        than the width/height of sectors
        Expected that it raises a ValueError
        """

        L = 10.0  # Width/height of simulation box

        bottom_left = [0.0, 0.0]
        top_right = [L, L]

        # Disc properties
        pos = [L/2, L/2]
        v = [0.0, 0.0]
        m = 1.0

        # First check x direction, sector width is 1.0 so R should be <= 0.5
        s = bl.PySim(bottom_left, top_right, 10, 1)

        with self.assertRaises(ValueError) as _:
            s.add_disc(pos, v, m, 0.6)

        # Now test vertical direction
        s = bl.PySim(bottom_left, top_right, 1, 10)

        with self.assertRaises(ValueError) as _:
            s.add_disc(pos, v, m, 0.6)




