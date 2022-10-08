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

def total_KE(m, v, I, w):
    ke = m * np.linalg.norm(v, axis=1)**2
    ke += I * w**2

    return np.sum(ke) / 2.0

def create_20_disc_sim(n_sector_x, n_sector_y, e_t, g):
    """
    Creates and runs a simulation containing 20 discs for verification pruposes
    Can choose the number of sectors to be used in the simulation
    """

    # Set the seed so we know add_random_discs() won't fail
    np.random.seed(30)

    box_bottom_left = [ 0.0,  0.0]
    box_top_right = [20.0, 20.0]

    s = bl.PySim(box_bottom_left, box_top_right, n_sector_x, n_sector_y)
    s.e_t = e_t
    s.g = g

    #s.add_box_walls(box_bottom_left, box_top_right)
    s.add_random_discs(np.array(box_bottom_left), np.array(box_top_right), 20, 1.0, 1.0, v=5.0)

    # Perform some collision
    s.advance(1000, 100.0, True)

    return {'box': [box_bottom_left, box_top_right], 's': s}

def test_overlapping(s):
    """
    Runs through a simulation s and tests if any discs overlap
    Expects the simulation has already been performed, i.e., advance() has 
    already been called on s
    """

    R = s.initial_state['R']

    for c_state in s.replay_by_time(0.1):
        pos = c_state['r']

        for i in range(pos.shape[0]):

            diff = pos[i+1:] - pos[i] 

            dist = np.linalg.norm(diff, axis=1)

            np.testing.assert_array_equal(R[i] + R[i+1:] <= dist, True)


class Test_PySim(unittest.TestCase):
    
    ####################################################################
    # Test collisions physics is corect for perfectly smooth discs
    ####################################################################
    def test_disc_wall_col_vertical_smooth_disc(self):
        """Tests smooth discs collide with vertical walls correctly"""
        s = bl.PySim([0.0, 0.0], [10.0, 10.0])

        s.add_disc([3.0, 3.0], [-1.0, 1.0], 1.0, 1.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([1.0, 5.0])
        expected_v = np.array([1.0, 1.0])

        self.assertAlmostEqual(ev.t, 2.0)
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 0)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)
        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v)

    def test_disc_wall_col_horizontal_smooth_disc(self):
        """Tests smooth discs collide with horizontal walls correctly"""
        s = bl.PySim([0.0, 0.0], [10.0, 10.0])

        s.add_disc([3.0, 7.0], [1.0, 1.0], 1.0, 1.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([5.0, 9.0])
        expected_v = np.array([1.0, -1.0])

        self.assertAlmostEqual(ev.t, 2.0)
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 1)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)
        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v)

    def test_disc_wall_col_diagonal_smooth_disc(self):
        """Tests smooth discs collide with diagonal walls correctly"""
        s = bl.PySim([-10.0, -10.0], [10.0, 10.0])

        s.add_wall([0.0, 0.0], [10.0, 10.0])
        s.add_disc([3.0, 0.0], [0.0, 1.0], 1.0, 1.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([3.0, 3.0 - 1 / np.sin(np.pi/4)])
        expected_v = np.array([1.0, 0.0])

        self.assertAlmostEqual(ev.t, 3.0 - 1 / np.sin(np.pi/4))
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 4)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)
        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v, atol=1e-15)

    def test_disc_disc_col_vertical_smooth_disc(self):
        """Tests smooth vertial disc-disc collisions"""
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([3.0, 0.0], [0.0, 1.0], 1.0, 1.0)
        s.add_disc([3.0, 10.0], [0.0, -1.0], 1.0, 1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([3.0, 4.0]),
                'v': np.array([0.0, -1.0]),
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([3.0, 6.0]),
                'v': np.array([0.0, 1.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])

    def test_disc_disc_col_horizontally_smooth_disc(self):
        """Tests smooth horizontal disc-dsic collisions"""
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([4.0, 0.0]),
                'v': np.array([-1.0, 0.0]),
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([6.0, 0.0]),
                'v': np.array([1.0, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
    
    def test_disc_disc_col_off_centre_smooth_disc(self):
        """
        Tests off centre disc-disc collisions for smooth discs, where disc 
        velocity isn't towards the other disc's centre. 
        """
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0)
        s.add_disc([10.0, np.sqrt(2)], [-1.0, 0.0], 1.0, 1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 5 - np.sqrt(2)/2,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([5 - np.sqrt(2)/2, 0.0]),
                'v': np.array([0.0, -1.0]),
            },
            {
                't': 5 - np.sqrt(2)/2,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([5 + np.sqrt(2)/2, np.sqrt(2)]),
                'v': np.array([0.0, 1.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'], atol=1e-14)

    def test_disc_disc_col_speed_smooth_disc(self):
        """
        Tests disc-disc collisions for smooth discs where each disc has a
        different speed
        """
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [2.0, 0.0], 1.0, 1.0)
        s.add_disc([10.0, 0.0], [-3.0, 0.0], 1.0, 1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 1.6,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([3.2, 0.0]),
                'v': np.array([-3.0, 0.0]),
            },
            {
                't': 1.6,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([5.2, 0.0]),
                'v': np.array([2.0, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
    
    def test_disc_disc_col_mass_smooth_disc(self):
        """
        Tests disc-disc collisions for smooth discs where each disc has a 
        different mass
        """
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 2.0, 1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 3.0, 1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([4.0, 0.0]),
                'v': np.array([-1.4, 0.0]),
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([6.0, 0.0]),
                'v': np.array([0.6, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
    
    def test_disc_disc_col_radii_smooth_disc(self):
        """
        Tests disc-disc collisions for smooth discs where each disc has a 
        different radius
        """
        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 2.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 3.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 2.5,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([2.5, 0.0]),
                'v': np.array([-1.0, 0.0]),
            },
            {
                't': 2.5,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([7.5, 0.0]),
                'v': np.array([1.0, 0.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
    
    def test_KE_conservation_smooth_disc(self):
        """
        Tests kinetic energy is approximately conserved during a 
        test simulation involving smooth discs
        """
        
        sim = create_20_disc_sim(1, 1, e_t=-1.0, g=[0.0, 0.0])  # Set e_t for smooth discs

        s = sim['s']

        initial_state = s.initial_state

        initial_KE = total_KE(initial_state['m'], initial_state['v'], initial_state['I'], initial_state['w'])

        for c_state in s.replay_by_time(0.1):
            m, v = c_state['m'], c_state['v']
            I, w = c_state['I'], c_state['w']

            current_KE = total_KE(m, v, I, w)

            self.assertAlmostEqual(current_KE, initial_KE)

    ####################################################################
    # Test collisions physics is corect for perfectly rough discs
    ####################################################################
    def test_disc_wall_col_vertical_rough_disc(self):
        """Tests rough discs collide with vertical walls correctly"""

        s = bl.PySim([0.0, 0.0], [10.0, 10.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([3.0, 3.0], [-1.0, 1.0], 1.0, 1.0, I=0.5, w=0.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([1.0, 5.0])
        expected_v = np.array([1.0, 1 / 3])
        expected_w = 4 / 3

        self.assertAlmostEqual(ev.t, 2.0)
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 0)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)

        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v)
        assert_allclose(ev.w, expected_w)

    def test_disc_wall_col_horizontal_rough_disc(self):
        """Tests rough discs collide with horizontal walls correctly"""

        s = bl.PySim([0.0, 0.0], [10.0, 10.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([3.0, 7.0], [1.0, 1.0], 1.0, 1.0, I=0.5, w=0.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([5.0, 9.0])
        expected_v = np.array([1 / 3, -1.0])
        expected_w = 4 / 3

        self.assertAlmostEqual(ev.t, 2.0)
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 1)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)
        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v)
        assert_allclose(ev.w, expected_w)

    def test_disc_wall_col_diagonal_rough_disc(self):
        """Tests rough discs collide with diagonal walls correctly"""

        s = bl.PySim([-10.0, -10.0], [10.0, 10.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_wall([0.0, 0.0], [10.0, 10.0])
        s.add_disc([3.0, 0.0], [0.0, 1.0], 1.0, 1.0, I=0.5, w=0.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([3.0, 3.0 - 1 / np.sin(np.pi/4)])
        expected_v = np.array([2 / 3, -1 / 3])
        expected_w = 2 * np.sqrt(2.0) / 3

        self.assertAlmostEqual(ev.t, 3.0 - 1 / np.sin(np.pi/4))
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 4)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)
        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v, atol=1e-15)
        assert_allclose(ev.w, expected_w, atol=1e-15)

    def test_disc_disc_col_vertical_rough_disc(self):
        """Tests rough vertial disc-disc collisions"""

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([3.0, 0.0], [0.0, 1.0], 1.0, 1.0, I=0.5, w=1.0)
        s.add_disc([3.0, 10.0], [0.0, -1.0], 1.0, 1.0, I=0.5, w=1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([3.0, 4.0]),
                'v': np.array([2 / 3, -1.0]),
                'w': - 1 / 3
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([3.0, 6.0]),
                'v': np.array([-2 / 3, 1.0]),
                'w': - 1 / 3
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
            assert_allclose(ev.w, exp['w'])

    def test_disc_disc_col_horizontally_rough_disc(self):
        """Tests rough horizontal disc-disc collisions"""

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0, I=0.5, w=1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 1.0, I=0.5, w=1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([4.0, 0.0]),
                'v': np.array([-1.0, -2 / 3]),
                'w': - 1 / 3
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([6.0, 0.0]),
                'v': np.array([1.0, 2 / 3]),
                'w': - 1 / 3
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
            assert_allclose(ev.w, exp['w'])
    
    def test_disc_disc_col_off_centre_rough_disc(self):
        """
        Tests off centre disc-disc collisions for rough discs, where disc 
        velocity isn't towards the other disc's centre. 
        """

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough
        
        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0, I=0.5, w=1.0)
        s.add_disc([10.0, np.sqrt(2)], [-1.0, 0.0], 1.0, 1.0, I=0.5, w=1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 5 - np.sqrt(2)/2,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([5 - np.sqrt(2)/2, 0.0]),
                'v': np.array([ (-1+np.sqrt(2)) / 3, -(2+np.sqrt(2)) / 3]),
                'w': (-1 + 2*np.sqrt(2)) / 3
            },
            {
                't': 5 - np.sqrt(2)/2,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([5 + np.sqrt(2)/2, np.sqrt(2)]),
                'v': np.array([(1 - np.sqrt(2)) / 3, (2+np.sqrt(2)) / 3 ]),
                'w': (-1 + 2*np.sqrt(2)) / 3
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'], atol=1e-14)
            assert_allclose(ev.w, exp['w'], atol=1e-15)

    def test_disc_disc_col_speed_rough_disc(self):
        """
        Tests disc-disc collisions for rough discs where each disc has a
        different speed
        """

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([0.0, 0.0], [2.0, 0.0], 1.0, 1.0, I=0.5, w=1.0)
        s.add_disc([10.0, 0.0], [-3.0, 0.0], 1.0, 1.0, I=0.5, w=1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 1.6,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([3.2, 0.0]),
                'v': np.array([-3.0, -2/3]),
                'w': -1/3
            },
            {
                't': 1.6,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([5.2, 0.0]),
                'v': np.array([2.0, 2/3]),
                'w': -1/3
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
            assert_allclose(ev.w, exp['w'])

    def test_disc_disc_col_mass_rough_disc(self):
        """
        Tests disc-disc collisions for rough discs where each disc has a 
        different mass
        """

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([0.0, 0.0], [1.0, 0.0], 2.0, 1.0, I=1.0, w=1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 3.0, 1.0, I=1.0, w=1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([4.0, 0.0]),
                'v': np.array([-1.4, -12 / 17]),
                'w': -7 / 17
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([6.0, 0.0]),
                'v': np.array([0.6, 8 / 17]),
                'w': -7 / 17
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
            assert_allclose(ev.w, exp['w'])
    
    def test_disc_disc_col_radii_rough_disc(self):
        """
        Tests disc-disc collisions for rough discs where each disc has a 
        different radius
        """

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 2.0, I=1.0, w=1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 3.0, I=1.0, w=1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 2.5,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([2.5, 0.0]),
                'v': np.array([-1.0, -2 / 3]),
                'w': -1 / 3
            },
            {
                't': 2.5,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([7.5, 0.0]),
                'v': np.array([1.0, 2 / 3]),
                'w': -1.0
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
            assert_allclose(ev.w, exp['w'])
    
    def test_disc_disc_col_moment_of_inertia_rough_disc(self):
        """
        Tests disc-disc collisions for rough discs where each disc has a 
        different moment of inertia
        """

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0, I=0.1, w=1.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 1.0, I=0.2, w=1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([4.0, 0.0]),
                'v': np.array([-1.0, -4 / 17]),
                'w': -23 / 17
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([6.0, 0.0]),
                'v': np.array([1.0, 4 / 17]),
                'w': -3 / 17
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
            assert_allclose(ev.w, exp['w'])
    
    def test_disc_disc_col_angular_velocity_rough_disc(self):
        """
        Tests disc-disc collisions for rough discs where each disc has a 
        different angular velocity
        """

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.e_t = 1.0  # Set discs to being perfectly rough

        s.add_disc([0.0, 0.0], [1.0, 0.0], 1.0, 1.0, I=0.5, w=2.0)
        s.add_disc([10.0, 0.0], [-1.0, 0.0], 1.0, 1.0, I=0.5, w=3.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 4.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([4.0, 0.0]),
                'v': np.array([-1.0, -5/3]),
                'w': -4/3
            },
            {
                't': 4.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([6.0, 0.0]),
                'v': np.array([1.0, 5/3]),
                'w': -1/3
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])
            assert_allclose(ev.w, exp['w'])
    
    def test_KE_conservation_rough_disc(self):
        """
        Tests kinetic energy is approximately conserved during a 
        test simulation involving rough discs
        """
        
        sim = create_20_disc_sim(1, 1, e_t=1.0, g=[0.0, 0.0])  # Set discs to being perfectly rough

        s = sim['s']

        initial_state = s.initial_state

        initial_KE = total_KE(initial_state['m'], initial_state['v'], initial_state['I'], initial_state['w'])

        for c_state in s.replay_by_time(0.1):
            m, v = c_state['m'], c_state['v']
            I, w = c_state['I'], c_state['w']

            current_KE = total_KE(m, v, I, w)

            self.assertAlmostEqual(current_KE, initial_KE)

    ####################################################################
    # Test collisions physics is corect for perfectly smooth discs
    # Under gravity
    ####################################################################

    def test_disc_wall_col_vertical_gravity(self):
        """Tests smooth discs collide with vertical walls correctly under gravity"""

        s = bl.PySim([0.0, 0.0], [10.0, 10.0])
        s.g = [0.0, -1.0]

        s.add_disc([3.0, 3.0], [-1.0, 1.0], 1.0, 1.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([1.0, 3.0])
        expected_v = np.array([1.0, -1.0])

        self.assertAlmostEqual(ev.t, 2.0)
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 0)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)
        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v)

    def test_disc_wall_col_horizontal_gravity(self):
        """Tests smooth discs collide with horizontal walls correctly under gravity"""

        s = bl.PySim([0.0, 0.0], [10.0, 10.0])
        s.g = [0.0, -1.0]

        s.add_disc([3.0, 2.0], [1.0, 1.0], 1.0, 1.0)

        s.advance(1, 10.0, True)

        # Check expected event properties
        ev = s.events[0]

        expected_pos = np.array([4.0+np.sqrt(3.0), 1.0])
        expected_v = np.array([1.0, np.sqrt(3.0)])

        self.assertAlmostEqual(ev.t, 1+np.sqrt(3.0))
        self.assertEqual(ev.ind, 0)
        self.assertEqual(ev.partner_ind, 3)
        self.assertEqual(ev.col_type, bl.PyCol_Type.Disc_Wall)
        assert_allclose(ev.r, expected_pos)
        assert_allclose(ev.v, expected_v)
        
    def test_disc_disc_col_gravity(self):
        """Tests smooth disc-disc collisions under gravity"""

        s = bl.PySim([-20.0, -20.0], [20.0, 20.0])
        s.g = [0.0, -1.0]

        s.add_disc([0.0, 0.0], [0.0, 0.0], 1.0, 1.0)
        s.add_disc([4.0, -2.0], [-1.0, 1.0], 1.0, 1.0)

        s.advance(2, 10.0, True)

        # Check expected event properties
        events = s.events

        if events[0].ind == 1:
            events[0], events[1] = events[1], events[0]
        
        expected = [
            {
                't': 2.0,
                'ind': 0,
                'partner_ind': 1,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([0.0, -2.0]),
                'v': np.array([-1.0, -2.0]),
            },
            {
                't': 2.0,
                'ind': 1,
                'partner_ind': 0,
                'col_type': bl.PyCol_Type.Disc_Disc,
                'r': np.array([2.0, -2.0]),
                'v': np.array([0.0, -1.0]),
            }
        ]

        for ev, exp in zip(events, expected):
            self.assertAlmostEqual(ev.t, exp['t'])
            self.assertEqual(ev.ind, exp['ind'])
            self.assertEqual(ev.partner_ind, exp['partner_ind'])
            self.assertEqual(ev.col_type, exp['col_type'])
            assert_allclose(ev.r, exp['r'])
            assert_allclose(ev.v, exp['v'])

    def test_energy_conservation_gravity(self):
        """
        Tests energy is approximately conserved during a 
        test simulation involving smooth discs & gravity
        """
        
        # Set e_t for smooth discs, gravity vertically downwards
        g = [0.0, -1.0]

        sim = create_20_disc_sim(1, 1, e_t=-1.0, g=g)  

        s = sim['s']

        initial_state = s.initial_state
        m = initial_state['m']

        initial_KE = total_KE(m, initial_state['v'], initial_state['I'], initial_state['w'])
        initial_PE = np.sum(m*np.linalg.norm(g)*initial_state['r'][:, 1])

        initial_energy = initial_KE + initial_PE

        for c_state in s.replay_by_time(0.1):
            m, v = c_state['m'], c_state['v']
            I, w = c_state['I'], c_state['w']

            y = c_state['r'][:, 1]  # altitude of discs

            current_KE = total_KE(m, v, I, w)
            current_PE = np.sum(m*np.linalg.norm(g)*c_state['r'][:, 1])

            current_energy = current_KE + current_PE

            self.assertAlmostEqual(current_energy, initial_energy)

    def test_overlapping_gravity_no_sectoring(self):
        """
        Tests discs don't overlap during a test simulation, with gravity, 
        no sectoring used
        """

        sim = create_20_disc_sim(1, 1, e_t=-1.0, g=[1.0, 1.0])  # No sectoring used, smooth discs
        s = sim['s']

        test_overlapping(s)
    
    def test_overlapping_gravity_sectoring(self):
        """
        Tests discs don't overlap during a test simulation, with gravity, 
        sectoring used
        """

        sim = create_20_disc_sim(9, 9, e_t=-1.0, g=[1.0, 1.0])  # sectoring used, smooth discs used
        s = sim['s']

        test_overlapping(s)

    def test_out_of_bounds_gravity(self):
        """
        Tests discs don't go out of bounds during a test simulation, gravity used
        """

        sim = create_20_disc_sim(1, 1, e_t=-1.0, g=[1.0, 1.0])  # Use smooth discs

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

    ####################################################################
    # Other tests
    ####################################################################
    def test_out_of_bounds_no_gravity(self):
        """
        Tests discs don't go out of bounds during a test simulation, no gravity
        """

        sim = create_20_disc_sim(1, 1, e_t=-1.0, g=[0.0, 0.0])  # Use smooth discs

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

    def test_overlapping_no_gravity_no_sectoring(self):
        """
        Tests discs don't overlap during a test simulation, no gravity, 
        no sectoring used
        """

        sim = create_20_disc_sim(1, 1, e_t=-1.0, g=[0.0, 0.0])  # No sectoring used, smooth discs
        s = sim['s']

        test_overlapping(s)
    
    def test_overlapping_no_gravity_sectoring(self):
        """
        Tests discs don't overlap during a test simulation, no gravity, 
        sectoring used
        """

        sim = create_20_disc_sim(9, 9, e_t=-1.0, g=[0.0, 0.0])  # sectoring used, smooth discs used
        s = sim['s']

        test_overlapping(s)

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
    
    def test_reject_adding_disc_invalid_moment_of_inerita(self):
        """
        Tests add_disc() rejects attempting to add a disc with a moment of 
        inertia that is negative/zero or is larger than m*R**2/2
        Expected that it raises a ValueError
        """

        L = 10.0  # Width/height of simulation box

        bottom_left = [0.0, 0.0]
        top_right = [L, L]

        # Disc properties, max moment of inertia is m*R**2/2 = 9
        pos = [L/2, L/2]
        v = [0.0, 0.0]
        m = 2.0
        R = 3.0

        s = bl.PySim(bottom_left, top_right, 1, 1)

        with self.assertRaises(ValueError) as _:  # negative I
            s.add_disc(pos, v, m, R, I=-1.0)

        with self.assertRaises(ValueError) as _:  # Zero I
            s.add_disc(pos, v, m, R, I=0.0)
        
        with self.assertRaises(ValueError) as _:  # Too large I
            s.add_disc(pos, v, m, R, I=1.01*m*R**2)

        # This should be allowed
        s.add_disc(pos, v, m, R, I=m*R**2 / 4)
        
    def test_reject_invalid_bounds(self):
        """
        Tests constructor for PySim rejects attempting to create a simulation 
        with invalid bounds, e.g. left bound is greater than or equal to right
        Expected that it raises a ValueError
        """

        L = 10.0  # Width/height of simulation box

        # First check right can't be less than left
        with self.assertRaises(ValueError) as _:
            bottom_left = [0.0, 0.0]
            top_right = [-L, L]

            s = bl.PySim(bottom_left, top_right, 10, 1)

        # Left can't be equal to right
        with self.assertRaises(ValueError) as _:
            bottom_left = [0.0, 0.0]
            top_right = [0.0, L]

            s = bl.PySim(bottom_left, top_right, 10, 1)

        # Bottom can't be greater than top
        with self.assertRaises(ValueError) as _:
            bottom_left = [0.0, 2*L]
            top_right = [L, L]

            s = bl.PySim(bottom_left, top_right, 10, 1)
        
        # Bottom can't be equal to top
        with self.assertRaises(ValueError) as _:
            bottom_left = [0.0, L]
            top_right = [L, L]

            s = bl.PySim(bottom_left, top_right, 10, 1)

    def test_reject_invalid_number_of_sectors(self):
        """
        Tests constructor for PySim rejects attempting to create a simulation 
        with invalid number of sectors, i.e. zero
        Expected that it raises a ValueError
        """

        L = 10.0  # Width/height of simulation box

        bottom_left = [0.0, 0.0]
        top_right = [L, L]

        # First check 0 sectors in x direction is not allowed
        with self.assertRaises(ValueError) as _:
            s = bl.PySim(bottom_left, top_right, 0, 1)

        with self.assertRaises(ValueError) as _:
            s = bl.PySim(bottom_left, top_right, 1, 0)

    def test_add_random_discs_bounds(self):
        """Tests that discs added are within the specified bounds"""

        L = 400.0  # Width/height of simulation box

        bottom_left = np.array([0.0, 0.0])
        top_right = np.array([L, L])

        # Masses & radii of disc
        m = 1.0
        R = 1.0

        # First check the random method
        s = bl.PySim(bottom_left, top_right, 1, 1)

        s.add_random_discs(bottom_left, top_right, 1_000, m, R, v=1.0, pos_allocation='random')

        pos = s.initial_state['r']

        self.assertTrue(all(bottom_left <= np.min(pos, axis=0)))
        self.assertTrue(all(np.max(pos, axis=0) <= top_right ))

        # Now check grid method of new PySim instance
        s = bl.PySim(bottom_left, top_right, 1, 1)

        s.add_random_discs(bottom_left, top_right, 1_000, m, R, v=1.0, pos_allocation='grid')

        pos = s.initial_state['r']

        self.assertTrue(all(bottom_left <= np.min(pos, axis=0)))
        self.assertTrue(all(np.max(pos, axis=0) <= top_right ))
    
    def test_add_random_discs_overlapping(self):
        """Tests that discs added are with no overlapping"""

        N_discs = 1_000
        L = 400.0  # Width/height of simulation box

        bottom_left = np.array([0.0, 0.0])
        top_right = np.array([L, L])

        # Masses & radii of disc
        m = 1.0
        R = 1.0

        # First check the random method
        s = bl.PySim(bottom_left, top_right, 1, 1)

        s.add_random_discs(bottom_left, top_right, N_discs, m, R, v=1.0, pos_allocation='random')

        pos = s.initial_state['r']

        for disc_ind in range(1, N_discs):
            dist = np.linalg.norm(pos[:disc_ind] - pos[disc_ind])

            self.assertTrue(np.all(dist >= 2*R))

        # Now check grid method
        # First check the random method
        s = bl.PySim(bottom_left, top_right, 1, 1)

        s.add_random_discs(bottom_left, top_right, N_discs, m, R, v=1.0, pos_allocation='grid')

        pos = s.initial_state['r']

        for disc_ind in range(1, N_discs):
            dist = np.linalg.norm(pos[:disc_ind] - pos[disc_ind])

            self.assertTrue(np.all(dist >= 2*R))

    def test_add_random_discs_temperature(self):
        """Tests that discs added do have the requested temperature"""

        N_discs = 10_000
        L = 1000.0  # Width/height of simulation box

        bottom_left = np.array([0.0, 0.0])
        top_right = np.array([L, L])

        # Masses & radii of disc
        m = 1.0
        R = 1.0
        kB_T = 1.0

        # First check the random method
        s = bl.PySim(bottom_left, top_right, 1, 1)

        s.add_random_discs(bottom_left, top_right, N_discs, m, R, kB_T=kB_T, pos_allocation='grid')

        # As a simple test, test the mean is about right
        # Mean should be sqrt(pi*kB*T/(2*m))

        speeds = np.linalg.norm(s.initial_state['v'], axis=1)

        speeds_mean = np.mean(speeds)
        expected_mean = np.sqrt(np.pi*kB_T/(2*m))

        assert_allclose(speeds_mean, expected_mean, atol=0.2)

    def test_e_n_allowed_values(self):
        """Tests PySim.e_n rejects values outside 0.0 <= e_n <= 1.0"""

        sim = bl.PySim(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        
        # All these values should be allowed
        sim.e_n = 1.0
        sim.e_n = 0.0
        sim.e_n = 0.5

        # All these should be rejected
        with self.assertRaises(ValueError) as _:
            sim.e_n = -0.1
        
        with self.assertRaises(ValueError) as _:
            sim.e_n = 1.1

    def test_e_t_allowed_values(self):
        """Tests PySim.e_t rejects values outside -1.0 <= e_t <= 1.0"""

        sim = bl.PySim(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        
        # All these values should be allowed
        sim.e_t = 1.0
        sim.e_t = -1.0
        sim.e_t = 0.0

        # All these should be rejected
        with self.assertRaises(ValueError) as _:
            sim.e_t = -1.1
        
        with self.assertRaises(ValueError) as _:
            sim.e_t = 1.1
    
    def test_replay_by_time_reject_negative_dt(self):
        """Tests PySim.replay_by_time() rejects dt<=0.0"""

        sim = create_20_disc_sim(1, 1, e_t=-1.0, g=[0.0, 0.0])  # Set e_t for smooth discs

        s = sim['s']

        with self.assertRaises(ValueError) as _:
            next(s.replay_by_time(0.0))  # generators don't execute until next is called
        
        with self.assertRaises(ValueError) as _:
            next(s.replay_by_time(-1.0))

