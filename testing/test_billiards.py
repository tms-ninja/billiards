import unittest

import numpy as np
from numpy.testing import assert_allclose

import billiards as bl

def total_KE(m, v):
    ke = m * np.linalg.norm(v, axis=1)**2

    return np.sum(ke) / 2.0

def create_20_disc_sim():
    """
    Creates and runs a simulation containing 20 discs for verification pruposes
    """

    # Set the seed so we know add_random_discs() won't fail
    np.random.seed(30)

    box_bottom_left = [ 0.0,  0.0]
    box_top_right = [20.0, 20.0]

    s = bl.PySim(box_bottom_left, box_top_right)

    #s.add_box_walls(box_bottom_left, box_top_right)
    s.add_random_discs(np.array(box_bottom_left), np.array(box_top_right), 20, 5.0, 1.0, 1.0)

    s.setup()

    # Perform some collision
    s.advance(1000, 100.0, True)

    return {'box': [box_bottom_left, box_top_right], 's': s}


class Test_PySim(unittest.TestCase):
    
    def test_disc_wall_col_vertical(self):
        """Tests discs collide with vertical walls correctly"""
        s = bl.PySim([0.0, 0.0], [0.0, 10.0])

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
        
        sim = create_20_disc_sim()

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

        sim = create_20_disc_sim()

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
        Tests discs don't overlap during a test simulation
        """

        sim = create_20_disc_sim()
        s = sim['s']

        R = s.initial_state['R']

        for c_state in s.replay_by_time(0.1):
            pos = c_state['r']

            for i in range(pos.shape[0]):

                diff = pos[i+1:] - pos[i] 

                dist = np.linalg.norm(diff, axis=1)

                np.testing.assert_array_equal(R[i] + R[i+1:] <= dist, True)



        
