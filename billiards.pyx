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

# distutils: language = c++


import enum
import math

from cython_header cimport *
cimport cython
from cpython.ref cimport PyObject
from libc.math cimport hypot

import numpy as np
cimport numpy as np

import scipy.stats

# Initialise numpy's C API
np.import_array()

cdef _get_state_pos(vector[Disc]& state):
    """
    Returns numpy array containing the position of every disc in state.
    
    Parameters
    ----------
    state : vector[Disc]&
        The state vector containing N discs.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N,2)
    """
    
    cdef size_t n_discs = state.size()

    arr = np.empty((n_discs, 2), dtype=np.float64)

    for d_ind in range(0, n_discs):
        for p_ind in range(0, 2):
            arr[d_ind, p_ind] = state[d_ind].r[p_ind]

    return arr

cdef _get_state_v(vector[Disc]& state):
    """
    Returns numpy array containing the velocity of every disc in state.
    
    Parameters
    ----------
    state : vector[Disc]&
        The state vector containing N discs.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N,2)
    """
    
    cdef size_t n_discs = state.size()

    arr = np.empty((n_discs, 2), dtype=np.float64)

    for d_ind in range(0, n_discs):
        for p_ind in range(0, 2):
            arr[d_ind, p_ind] = state[d_ind].v[p_ind]

    return arr

cdef _get_state_w(vector[Disc]& state):
    """
    Returns numpy array containing the angular velocity of every disc in state.
    
    Parameters
    ----------
    state : vector[Disc]&
        The state vector containing N discs.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N,)
    """
    
    cdef size_t n_discs = state.size()

    arr = np.empty((n_discs, ), dtype=np.float64)

    for d_ind in range(0, n_discs):
        arr[d_ind] = state[d_ind].w

    return arr

cdef _get_state_m(vector[Disc]& state):
    """
    Returns numpy array containing the mass of every disc in state.
    
    Parameters
    ----------
    state : vector[Disc]&
        The state vector containing N discs.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N)
    """
    
    cdef size_t n_discs = state.size()

    arr = np.empty((n_discs, ), dtype=np.float64)

    for d_ind in range(0, n_discs):
        arr[d_ind] = state[d_ind].m

    return arr

cdef _get_state_R(vector[Disc]& state):
    """
    Returns numpy array containing the radius of every disc in state.
    
    Parameters
    ----------
    state : vector[Disc]&
        The state vector containing N discs.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N)
    """
    
    cdef size_t n_discs = state.size()

    arr = np.empty((n_discs, ), dtype=np.float64)

    for d_ind in range(0, n_discs):
        arr[d_ind] = state[d_ind].R

    return arr

cdef _get_state_I(vector[Disc]& state):
    """
    Returns numpy array containing the moment of inertia of every disc in state.
    
    Parameters
    ----------
    state : vector[Disc]&
        The state vector containing N discs.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N,)
    """
    
    cdef size_t n_discs = state.size()

    arr = np.empty((n_discs, ), dtype=np.float64)

    for d_ind in range(0, n_discs):
        arr[d_ind] = state[d_ind].I

    return arr


cdef _test_collisions(np.ndarray pos, np.ndarray radii, double max_R=-1.0):
    """
    Tests for collisions between last position and all earlier positions
    Returns True if there are no collisions between discs.

    Parameters
    ----------
    pos : np.ndarray
        The positions of the centres of all discs. The position of the disc to
        be tested should be at pos[-1]. Expected to have shape (N, 2).
    radii : np.ndarray
        The radii of the discs. The radius of the disc to be tested should be 
        at radii[-1]. Expected to have shape (N, 2).
    max_R : double, optional
        The maximum radius any disc has. If not set or set to -1.0, 
        _test_collision() will determine the maximum. The default it -1.0.

    Returns
    -------
    bool
        Returns True if there are no collision, False otherwise.
    """

    if pos.shape[0]==1:
        return True

    cdef double[:, :] pos_view = pos
    cdef double[:] radii_view = radii

    cdef double R_disc = radii_view[-1]
    cdef double max_closest_dist

    if max_R==-1.0:
        max_closest_dist = R_disc + np.max(radii[:-1])
    else:
        max_closest_dist = R_disc + max_R

    # Use sweep and prune to reduce number of discs that need checking
    # Sort discs by x position
    cdef np.ndarray pos_x = pos[:-1, 0]

    # Use NumPy's C API as the overhead calling through Python is very significant (2-3 times faster)
    cdef np.ndarray sorted_args = np.PyArray_ArgSort(pos_x, 0, np.NPY_QUICKSORT)  # arg sort along axis 0

    cdef PyObject* sorted_args_ptr = <PyObject*>sorted_args
    cdef double new_disc_x = pos_view[-1, 0]

    # left_ind includes the one we need to check
    cdef int left_ind = np.PyArray_SearchSorted(pos_x, float(new_disc_x-max_closest_dist), np.NPY_SEARCHLEFT, sorted_args_ptr)

    # right_ind is 1 past what we need to check
    cdef int right_ind = np.PyArray_SearchSorted(pos_x, float(new_disc_x+max_closest_dist), np.NPY_SEARCHRIGHT, sorted_args_ptr)
    
    cdef int i
    cdef int disc_to_check

    cdef double[2] diff_pos

    for i in range(left_ind, right_ind):
        disc_to_check = sorted_args[i]

        for coord_ind in range(2):
            diff_pos[coord_ind] = pos_view[-1, coord_ind] - pos_view[disc_to_check, coord_ind]

        if hypot(diff_pos[0], diff_pos[1]) < R_disc + radii_view[disc_to_check]:
            return False
    
    return True

# class PyEvent

# Wrapping for C++ Collision_Type
class PyCol_Type(enum.Enum):
    """Enum to describe the type of event that occured, e.g. disc-disc collision"""
    Disc_Disc = 0
    Disc_Wall = 1
    Disc_Boundary = 2
    Disc_Advancement = 3


cdef class PyEvent():
    """Represents a collision event of a disc"""

    # We hold onto the PySim instance and use an index to avoid memory
    # management issues that may arise when using a pointer
    cdef PySim _sim
    cdef size_t _e_ind

    def __repr__(self):
        """String representation of PyEvent"""

        rep = f"PyEvent(t={self.t}, ind={self.ind}, partner_ind={self.partner_ind}, col_type={self.col_type}, "
        rep += f"r={self.r}, v={self.v}, w={self.w})"
        
        return rep
    
    @staticmethod
    cdef from_PySim(PySim s, size_t ind):
        """
        Creates a PyEvent instance from the Event at index ind in the given
        simulation

        Parameters
        ----------
        s : PySim
            The simulation that contains the list of events this event wil be 
            created from.
        ind : int
            The index in the events vector which this instance of PyEvent will
            wrap.
        
        Returns
        -------
        PyEvent
            The PyEvent instance that wraps the event at ind.

        """

        cdef PyEvent e = PyEvent()

        e._sim = s
        e._e_ind = ind  # index of event in C++ events vector

        return e

    # Properties
    @property
    def t(self):
        """
        Time of collision
        
        Parameters
        ----------
        None.

        Returns
        -------
        float
            Returns the time the collision occured at.

        """

        return self._sim.s.events[self._e_ind].t

    @property
    def ind(self):
        """
        Index of the collising disc

        Parameters
        ----------
        None.

        Returns
        -------
        int
            Returns the index of the colliding disc.
            
        """

        return self._sim.s.events[self._e_ind].ind

    @property
    def partner_ind(self):
        """
        Index of partner object (e.g. disc or wall) involved in the collision
        
        Parameters
        ----------
        None.

        Returns
        -------
        int
            Returns the index of the secondary object
            
        """

        return self._sim.s.events[self._e_ind].partner_ind

    @property
    def col_type(self):
        """
        Indicates whether the collision involved a wall or another disc
        
        Parameters
        ----------
        None.

        Returns
        -------
        PyCol_Type
            Returns an instance of PyCol_Type describing the nature of the 
            collision
            
        """

        cdef Collision_Type col_type = self._sim.s.events[self._e_ind].col_type

        if col_type==Disc_Disc:
            return PyCol_Type.Disc_Disc
        elif col_type==Disc_Wall:
            return PyCol_Type.Disc_Wall
        elif col_type==Disc_Boundary:
            return PyCol_Type.Disc_Boundary
        else:
            return PyCol_Type.Disc_Advancement


    @property
    def r(self):
        """
        Position of disc at time of collision
        
        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            Returns a numpy array with shape (2,) of the position of the disc
            
        """

        n = np.empty((2,), dtype=np.float64)

        for i in range(2):
            n[i] = self._sim.s.events[self._e_ind].r[i]

        return n

    @property
    def v(self):
        """
        Velocity of disc after collision
        
        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            Returns a numpy array with shape (2,) of the new velocity
            
        """

        n = np.empty((2,), dtype=np.float64)

        for i in range(2):
            n[i] = self._sim.s.events[self._e_ind].v[i]

        return n
    
    @property
    def w(self):
        """
        Angular velocity of disc after collision
        
        Parameters
        ----------
        None.

        Returns
        -------
        float
            Returns the new angular velocity of the disc
            
        """

        return self._sim.s.events[self._e_ind].w

# class PySim

cdef class PySim():
    """
    Represents a simulation of discs, not intended to be initialized directly
    """

    cdef Sim* s
    cdef list _events

    def __init__(self, bottom_left, top_right, size_t N=1, size_t M=1):
        """
        Initilizes the PySim instance
        
        Parameters
        ----------
        bottom_left : numpy.ndarray or list
            The bottom left corner of the box the simulation takes place in. 
            Expected to be a numpy array with shape (2, ) or a list with length
            2.
        top_right : numpy.ndarray or list
            The top right corner of the box the simulation takes place in. 
            Expected to be a numpy array with shape (2, ) or a list with length
            2.
        N : int
            Number of sectors the simulation box is split into in the horizontal
            direction. This only includes sectors within the simulation box 
            boundaries.
        M : int
            Number of sectors the simulation box is split into in the vertical
            direction. This only includes sectors within the simulation box
            boundaries.

        Raises
        ------
        ValueError
            Raised if the top-right corner isn't above and to the right of the 
            bottom-left corner or the number of sectors in either direction is
            zero.

        Returns
        -------
        None.

        """

        cdef double left, right, bottom, top

        left = bottom_left[0]
        bottom = bottom_left[1]
        right = top_right[0]
        top = top_right[1]

        cdef Vec2D v_bottom_left, v_top_right

        v_bottom_left = Vec2D(left, bottom)
        v_top_right = Vec2D(right, top)

        self.s = new Sim(v_bottom_left, v_top_right, N, M)


        self._events = []

    def __dealloc__(self):
        """
        Deallocates the PySim instance
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        del self.s

    def advance(self, size_t max_iterations, double max_t, bool record_events):
        """
        Advances the simulation by either max_iterations iterations or max_t
        duration, whichever is reached sooner. record events specifies whether
        events should be recorded or not.

        Parameters
        ----------
        max_iterations : int
            The maximum number of collisions to be performed before stopping.
            It may not be reached if the simulation time reaches max_t first.
        max_t : float
            The maximum duration of the simulation, when reached the simulation
            will be stopped. If max_iterations collisions are processed, max_t
            may not be reached.
        record_events : bool
            If True, the simulation's events will be recorded. Otherwise they
            will not.

        Returns
        -------
        None.

        """

        self.s.advance(max_iterations, max_t, record_events)

        # Need to add new events to the events list
        for ind in range(len(self._events), self.s.events.size()):
            self._events.append(PyEvent.from_PySim(self, ind))

    def add_wall(self, start, end):
        """
        Add a wall to the simulation with start point start and end point end.

        Parameters
        ----------
        start : numpy.ndarray
            The start point of the wall, expected to be a numpy array with 
            shape (2,).
        end : numpy.ndarray
            The end point of the wall, expected to be a numpy array with shape
            (2,).

        Returns
        -------
        None.

        """

        cdef Vec2D v_start = Vec2D(start[0], start[1])
        cdef Vec2D v_end = Vec2D(end[0], end[1])
        cdef Wall w = Wall(v_start, v_end)

        self.s.walls.push_back(w)

    def add_box_walls(self, bottom_left, top_right):
        """
        Adds four walls in the shape of a box, defined by its bottom left and
        top right corners.

        Parameters
        ----------
        bottom_left : numpy.ndarray or list
            The bottom left corner of the box. Expected to be a numpy array of 
            shape (2,) or a list of length 2.
        top_right : numpy.ndarray or list
            The top right corner of the box. Expected to be a numpy array of 
            shape (2,) or a list of length 2. 

        Returns
        -------
        None.

        """

        left = bottom_left[0]
        bottom = bottom_left[1]
        right = top_right[0]
        top = top_right[1]

        self.add_wall([left, bottom], [left, top])
        self.add_wall([left, top], [right, top])
        self.add_wall([right, top], [right, bottom])
        self.add_wall([right, bottom], [left, bottom])

    def add_disc(self, r, v, double m, double R, I=None, w=0.0):
        """
        Add a disc to the simulation with initial position r, velocity v, mass m
        and radius R.

        Parameters
        ----------
        r : numpy.ndarray
            The initial position of the disc, expected to be a numpy array with
            shape (2,).
        v : numpy.ndarray
            The initial velocity of the disc, expected to be a numpy array with
            shape (2,).
        m : float
            The mass of the disc.
        R : float
            The radius of the disc.
        I : float
            The moment of inertia of the disc. If set to None it will be 
            calculated assuming the disc is uniform, i.e. (m*R**2) / 2, The 
            default is None.
        w : float
            The angular velocity of the disc. The default is zero.

        Raises
        ------
        ValueError
            Raised if 
                - disc is not within the bounds of the simulation
                - mass/radius/moment of inertia of disc are less than or equal 
                  to zero
                - disc is larger than a sector in the simulation
                - moment of inertia is larger than maximum physical

        Returns
        -------
        None.

        """

        cdef Vec2D v_r = Vec2D(r[0], r[1])
        cdef Vec2D v_v = Vec2D(v[0], v[1])

        cdef double _I = m*R**2 / 2.0 if I is None else I 

        self.s.add_disc(v_r, v_v, w, m, R, _I)

    def add_random_discs(self, bottom_left, top_right, N_discs, m, R, v=None, kB_T=None,
                         pos_allocation='random'):
        """
        Adds N_discs discs with random positions and velocity directions in a 
        box defined by its bottom left and top right corners. Discs will be 
        completely contained within this box.

        Parameters
        ----------
        bottom_left : numpy.ndarray
            The bottom left corner of the box, expected to have shape (2,).
        top_right : numpy.ndarray
            The top right corner of the box, expected to have shape (2,).
        N_discs : int
            The number of discs that will be added.
        m : float or numpy.ndarray
            The mass added discs will have. If a numpy array is passed, it 
            should have shape (N_discs,).
        R : float or numpy.ndarray
            The radius added discs will have. If a numpy array is passed, it 
            should have shape (N_discs,).
        v : float, numpy.ndarray or None, optional
            The speed added discs will have. If a numpy array is passed, it 
            should have shape (N_discs,). If it is None, kB_T should be specified.
            The default is None.
        kB_T : float or None, optional
            The temperature added discs will have, drawn from a 2d 
            Maxwell-Boltzmann distribution. If it is None, v should be specified.
            The default is None.
        pos_allocation : str, optional
            The method used to allocate the positions of the discs. If 'random'
            is used, add_random_discs() tries to select a random position 
            within the box up to 10 times to prevent overlapping. If it fails 
            after 10 attempts, add_random_discs() fails. If 'grid' is selected,
            discs will be randomly allocated a place on a square grid. A 
            maximum of 10 attempts will be made before add_random_discs() 
            fails. Note the grid is square with respect to the number of discs 
            in the x/y directions, not the physical extent. The default is 
            'random'.

        Returns
        -------
        None.

        Raises
        ------
        RuntimeError
            Raised if it fails to place all discs. No new discs are added to 
            the simulation if this is raised.
        
        """

        current_state = self.current_state

        N_current_state = current_state['m'].shape[0]

        mass = np.empty(N_current_state + N_discs, dtype=np.float64)

        mass[:N_current_state] = current_state['m']
        mass[N_current_state:] = m

        radius = np.empty(N_current_state + N_discs, dtype=np.float64)

        radius[:N_current_state] = current_state['R']
        radius[N_current_state:] = R

        # Now set about generating velocities of discs
        if v is None and kB_T is None:
            raise ValueError("One of v or kB_T must be specified, both were None")
        elif v is not None and kB_T is not None:
            raise ValueError(f"Only one of v and T can be specified. v was: {v}, kB_T was: {kB_T}")

        velocity = np.empty((N_current_state + N_discs, 2), dtype=np.float64)
        angle = 2*np.pi*np.random.random(N_discs)
        velocity[:N_current_state] = current_state['v']

        # All new discs either have the passed speed(s) 
        if v is not None:
            velocity[N_current_state:, 0] = v*np.cos(angle)
            velocity[N_current_state:, 1] = v*np.sin(angle)
        else:
            # Need to draw from a 2d Maxwell-Boltzmann distribution
            # need to be careful if not all discs have the same mass, initially 
            # we'll reject this possibility
            unique_masses = np.unique(m)

            if unique_masses.shape[0]!=1:
                raise ValueError("Currently, all particles must have the same mass if kB_T is defined")

            scale = np.sqrt(kB_T / unique_masses[0])

            speeds = scipy.stats.chi.rvs(2, loc=0, scale=scale, size=N_discs)

            velocity[N_current_state:, 0] = speeds*np.cos(angle)
            velocity[N_current_state:, 1] = speeds*np.sin(angle)

        # Disc centres must be in a smaller box so they don't intersect the
        # walls
        pos = np.empty((N_current_state + N_discs, 2), dtype=np.float64)
        pos[:N_current_state] = current_state['r']

        if pos_allocation=='random':            
            pos = self._add_random_disc_random(N_discs, np.array(bottom_left), np.array(top_right), radius)
        elif pos_allocation=='grid':
            pos = self._add_random_disc_grid(N_discs, bottom_left, top_right, radius)
        else:
            raise ValueError(f"Unknown pos_allocation: {pos_allocation}. Allowed values are 'random' or 'grid'.")

        # Now add the discs to the simulation
        for ind in range(N_current_state, N_current_state + N_discs):
            self.add_disc(pos[ind], velocity[ind], mass[ind], radius[ind]) 

    def _add_random_disc_random(self, int N_discs, np.ndarray bottom_left, np.ndarray top_right, np.ndarray radius):
        """
        Computes positions for new discs using 'random' method
        
        Parameters
        ----------
        N_discs : int
            Number of discs to add.
        bottom_left : np.ndarray
            Bottom left corner of the box to add discs into.
        top_right : np.ndarray
            Top right corner of the box to add discs into.
        radius : np.ndarray
            Radii of the discs. Expected to have shape (N_discs,).

        Raises
        ------
        RuntimeError
            Raised if it failed to place a disc after 50 attempts. No new discs
            are added to the simulation if this is raised.

        Returns
        -------
        np.ndarray
            An np.ndarray of shape (MN_discs, 2) with the positions of the 
            discs, M is the number of previously added discs.
        """

        cdef int N_current_state = self.current_state['m'].shape[0]

        cdef np.ndarray pos = np.empty((N_current_state + N_discs, 2), dtype=np.float64)
        pos[:N_current_state] = self.current_state['r']

        cdef double[:, :] pos_view = pos

        cdef int max_attempt = 50
        cdef int attempt
        cdef int d_ind, coord_ind

        cdef double[:] radius_view = radius
        cdef double max_R = np.max(radius)
        cdef double R

        cdef double[:] bottom_left_view = bottom_left
        cdef double[:] top_right_view = top_right

        cdef double[2] _bottom_left
        cdef double[2] _top_right
        cdef double[2] diff

        # TODO Avoid caching random numbers
        # Currently we cache N_random random numbers to reduce Python calles
        # to NumPy. Should probably find a better way fo doing this
        cdef int N_random = 1_000
        cdef int next_random_ind = 0
        cdef np.ndarray random_value_array = np.random.random(N_random)
        cdef double random_value

        for d_ind in range(N_current_state, N_current_state + N_discs):
            attempt = max_attempt

            R = radius_view[d_ind]

            for coord_ind in range(2):
                _bottom_left[coord_ind] = bottom_left_view[coord_ind] + R
                _top_right[coord_ind] = top_right_view[coord_ind] - R

                diff[coord_ind] = _top_right[coord_ind] - _bottom_left[coord_ind]

            pos_sliced = pos[:d_ind+1]
            radius_sliced = radius[:d_ind+1]

            while attempt > 0:
                for coord_ind in range(2):
                    if next_random_ind==N_random:
                        np.copyto(random_value_array, np.random.random(N_random))
                        next_random_ind = 0
                    
                    random_value = random_value_array[next_random_ind]
                    next_random_ind += 1

                    pos_view[d_ind, coord_ind] = _bottom_left[coord_ind] + diff[coord_ind] *random_value

                if _test_collisions(pos_sliced, radius_sliced, max_R):
                    break  # No collisions, can move onto next disc

                # Disc does have a collision, try again
                attempt -= 1
            else:
                raise RuntimeError(f"Unable to place disc {d_ind - N_current_state} after {max_attempt} attempts.")
        
        return pos

    def _add_random_disc_grid(self, N_discs, bottom_left, top_right, radius):
        """
        Computes positions for new discs using 'grid' method
        
        Parameters
        ----------
        N_discs : int
            Number of discs to add.
        bottom_left : np.ndarray
            Bottom left corner of the box to add discs into.
        top_right : np.ndarray
            Top right corner of the box to add discs into.
        radius : np.ndarray
            Radii of the discs. Expected to have shape (N_discs,).

        Raises
        ------
        RuntimeError
            Raised if it failed to place a disc after 50 attempts. No new discs
            are added to the simulation if this is raised.

        Returns
        -------
        np.ndarray
            An np.ndarray of shape (MN_discs, 2) with the positions of the 
            discs, M is the number of previously added discs.
        """
        
        N_current_state = self.current_state['m'].shape[0]

        pos = np.empty((N_current_state + N_discs, 2), dtype=np.float64)
        pos[:N_current_state] = self.current_state['r']

        # Small margin so discs won't be touching bounds of requested box
        R_max = np.max(radius[N_current_state:])
        _bottom_left = bottom_left + R_max*1.001
        _top_right = top_right - R_max*1.001
        
        # require n_per_side**2 >= N _discs
        n_per_side = 1 + math.isqrt(N_discs-1)

        x_pos = np.linspace(_bottom_left[0], _top_right[0], n_per_side)
        y_pos = np.linspace(_bottom_left[1], _top_right[1], n_per_side)

        xx, yy = np.meshgrid(x_pos, y_pos)

        # In general we expect n_per_side**2 > N_discs, so we randomly 
        # choose which positions on the grid to use
        possible_positions = np.column_stack((xx.ravel(), yy.ravel()))

        attempt = 0
        max_attempt = 10

        while attempt < max_attempt:
            # select the positions for this attempt
            selected_indices = np.random.choice(possible_positions.shape[0], N_discs, replace=False)
            pos[N_current_state:] = possible_positions[selected_indices]

            overlapping_discs = False

            # Now check there aren't any discs overlapping
            if n_per_side > 1:
                dx, dy = x_pos[1] - x_pos[0], y_pos[1] - y_pos[0]

                max_R = np.max(radius)

                if dx > 2*max_R and dy > 2*max_R and N_current_state==0:
                    # Guaranteed there are no overlapping discs
                    break
                else:
                    for d_ind in range(N_current_state, N_current_state + N_discs):
                        #d_pos = pos[d_ind]

                        if not _test_collisions(pos[:d_ind+1], radius[:d_ind+1]):
                            overlapping_discs = True
                            break

            # we've allocated positions to all discs without any overlapping
            if not overlapping_discs:
                break

            attempt += 1
        else:
            raise RuntimeError(f"Unable to place discs on grid after {max_attempt} attempts")
        
        return pos

    # Properties
    @property
    def events(self):
        """
        Returns a list of events that occurred during the simulation
        
        Parameters
        ----------
        None.

        Returns
        -------
        list of PyEvent
            List of PyEvents
        """

        return self._events

    @property
    def initial_state(self):
        """
        Gets desired inital properties of each disc in the initial state and 
        returns them as a dictionary of numpy arrays.
        
        Parameters
        ----------
        None.

        Returns
        -------
        dict
            A dictionary with character keys and values of numpy arrays. Keys
            correspond to:
            - 'r' - position
            - 'v' - velocity
            - 'w' - angular velocity
            - 'm' - mass
            - 'R' - radius
            - 'I' - moment of inertia
        """

        state_dict = {}

        state_dict['r'] = _get_state_pos(self.s.initial_state)
        state_dict['v'] = _get_state_v(self.s.initial_state)
        state_dict['w'] = _get_state_w(self.s.initial_state)
        state_dict['m'] = _get_state_m(self.s.initial_state)
        state_dict['R'] = _get_state_R(self.s.initial_state)
        state_dict['I'] = _get_state_I(self.s.initial_state)

        return state_dict
    
    @property
    def current_state(self):
        """
        Gets desired current properties of each disc in the current state and 
        returns them as a dictionary of numpy arrays.
        
        Parameters
        ----------
        None.

        Returns
        -------
        dict
            A dictionary with character keys and values of numpy arrays. Keys
            correspond to:
            - 'r' - position
            - 'v' - velocity
            - 'w' - angular velocity
            - 'm' - mass
            - 'R' - radius
            - 'I' - moment of inertia
        """

        state_dict = {}

        state_dict['r'] = np.empty((self.s.initial_state.size(), 2), dtype=np.float64)
        state_dict['v'] = np.empty((self.s.initial_state.size(), 2), dtype=np.float64)
        state_dict['w'] = np.empty((self.s.initial_state.size(), ), dtype=np.float64)

        # Assume mass/radii/moment of inertia of discs don't change during simulation
        state_dict['m'] = _get_state_m(self.s.initial_state)
        state_dict['R'] = _get_state_R(self.s.initial_state)
        state_dict['I'] = _get_state_I(self.s.initial_state)

        cdef Event cur_disc


        for d_ind in range(0, self.s.initial_state.size()):
            cur_disc = self.s.events_vec[d_ind][self.s.old_vec[d_ind]]

            for p_ind in range(0, 2):
                state_dict['r'][d_ind, p_ind] = cur_disc.r[p_ind]
                state_dict['v'][d_ind, p_ind] = cur_disc.v[p_ind]
                
            state_dict['w'][d_ind] = cur_disc.w

        return state_dict

    @property
    def bounds(self):
        """
        Returns the bounds of the simulation.

        Parameters
        ----------
        None.

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing the coordinates of the bottom left and top right
            corners.
         
        """

        cdef double left, right, bottom, top

        left = self.s.bottom_left[0]
        bottom = self.s.bottom_left[1]
        right = self.s.top_right[0]
        top = self.s.top_right[1]

        bottom_left = np.array([left, bottom])
        top_right = np.array([right, top])

        return (bottom_left, top_right)

    @property
    def e_n(self):
        """
        Gets or sets the coefficient of normal restitution. Should be between
        0.0 and 1.0 inclusive.

        Raises
        ------
        ValueError
            Raised if one attempts to set e_n > 1.0 or e_n < 0.0
        
        """

        return self.s.get_e_n()
    @e_n.setter
    def e_n(self, double new_e_n):
        self.s.set_e_n(new_e_n)
    
    @property
    def e_t(self):
        """
        Gets or sets the coefficient of tangential restitution. Should be 
        between -1.0 and 1.0 inclusive. -1.0 corresponds to perfectly smooth
        discs, +1.0 to perfectly rough discs.

        Raises
        ------
        ValueError
            Raised if one attempts to set e_t > 1.0 or e_t < -1.0
        
        """

        return self.s.get_e_t()
    @e_t.setter
    def e_t(self, double new_e_t):
        self.s.set_e_t(new_e_t)

    @property
    def g(self):
        """
        Gets or sets the acceleration due to gravity each disc in the 
        simulation experiences. Note it should not be altered after advance()
        is called for the first time.
        """

        cdef Vec2D g_vec = self.s.get_g()
        cdef np.ndarray g_np = np.array([g_vec[0], g_vec[1]])

        return g_np
    @g.setter
    def g(self, new_g):
        cdef np.ndarray _g = np.array(new_g, dtype=np.float64)

        cdef Vec2D g_vec = Vec2D(_g[0], _g[1])

        self.s.set_g(g_vec)
    
    @property
    def current_time(self):
        """
        Gets the time the simulation has been advanced to. All collisions up to
        but not necessarily including current_time have been processed, though
        discs may not have been advanced and therefore have the properties 
        (i.e. position & velocity) they should have at current_time.
        """

        return self.s.current_time

    # Generators for replaying the simulation
    def replay_by_event(self):
        """
        Generator that yields a state dict updated event by event, starting 
        with the initial state. Each yield corresponds to advancing the 
        simulation by a single event. The two events associated with disc-disc
        collisions are yielded seperately, meaning two yields are needed to
        process a disc-disc collision.

        Parameters
        ----------
        None.

        Yields
        ------
        dict
            A dict containing the position, velocity, masses and radii of the 
            discs as numpy arrays. Keys
            correspond to:
            - 'r' - position
            - 'v' - velocity
            - 'w' - angular velocity
            - 'm' - mass
            - 'R' - radius
            - 'I' - moment of inertia
        
        """

        cdef size_t N_events = self.s.events.size()
        cdef size_t cur_disc
        cdef double current_t = 0.0
        cdef double dt

        current_state = self.initial_state

        yield current_state

        for current_state_ind in range(N_events):
            dt = self.s.events[current_state_ind].t - current_t
            cur_disc = self.s.events[current_state_ind].ind

            # Advance all discs to time of collision
            current_state['r'] += dt*current_state['v'] + self.g*(dt**2/2.0)
            current_state['v'] += self.g*dt


            # Update the colliding particle accordingly
            for ind in range(2):
                current_state['v'][cur_disc][ind] = self.s.events[current_state_ind].v[ind]

            # Update angular velocity
            current_state['w'][cur_disc] = self.s.events[current_state_ind].w

            current_t = self.s.events[current_state_ind].t

            yield current_state

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def replay_by_time(self, double dt):
        """
        Replays the simulation advancing by timesteps dt. If the final event 
        is not an integer multiple of dt, the events after the previous integer
        multiple of dt are not returned. I.e. If the last event in the 
        simulation occurs at time T, replay_by_time() does not consider events
        after int(T/dt).

        Parameters
        dt : float
            The timestep the replay advances by with each yield.

        Yields
        ------
        dict
            A dict containing the position, velocity, masses and radii of the 
            discs as numpy arrays. Keys
            correspond to:
            - 'r' - position
            - 'v' - velocity
            - 'w' - angular velocity
            - 'm' - mass
            - 'R' - radius
            - 'I' - moment of inertia
        
        Raises
        ------
        ValueError
            Raised if dt is less than or equal to zero
        
        """

        # Ensure dt is greater than zero
        if dt<=0.0:
            raise ValueError(f"dt must be greater than zero. dt was {dt}")

        current_state = self.initial_state

        yield current_state

        cdef double[:] g_view = self.g

        cdef size_t N_events = self.s.events.size()
        cdef int N_discs = current_state['r'].shape[0]
        cdef size_t cur_event_ind = 0
        cdef size_t cur_disc_ind

        # Start time for the current period
        cdef int cur_period = 1

        # current time each disc its current position, velocity etc. is valid for
        cdef np.ndarray current_t = np.zeros(N_discs)
        cdef double time_step        # time step used when advancing a single disc

        cdef double[:] current_t_view = current_t

        cdef double[:, :] r_view = current_state['r']
        cdef double[:, :] v_view = current_state['v']
        cdef double[:] w_view = current_state['w']

        # The inner loop corresponds to advancing one event at a time, the 
        # outer loop corresponds to advancing by dt
        while True:

            while cur_event_ind < N_events:
                # Next event is in the current period
                if self.s.events[cur_event_ind].t <= cur_period*dt:

                    cur_disc_ind = self.s.events[cur_event_ind].ind

                    # advance to next event
                    time_step = self.s.events[cur_event_ind].t - current_t_view[cur_disc_ind]
                    current_t_view[cur_disc_ind] = self.s.events[cur_event_ind].t

                    # Update disc position, velocity & angular velocity
                    for ind in range(2):
                        r_view[cur_disc_ind, ind] += time_step*v_view[cur_disc_ind, ind] + g_view[ind]*(time_step**2)/2.0

                    for ind in range(2):
                         v_view[cur_disc_ind, ind] = self.s.events[cur_event_ind].v[ind]

                    w_view[cur_disc_ind] = self.s.events[cur_event_ind].w
                else:
                    # advance to end of time interval and break
                    for disc_ind in range(N_discs):
                        time_step = cur_period*dt - current_t_view[disc_ind]

                        for ind in range(2):
                            r_view[disc_ind, ind] += time_step*v_view[disc_ind, ind] + (g_view[ind]/2.0)*time_step*time_step
                            v_view[disc_ind, ind] += g_view[ind]*time_step

                    # Seems more performant to let numpy set current t, possibly due to SIMD
                    current_t[...] = cur_period*dt

                    break

                cur_event_ind += 1
            
            # No more events to process
            if cur_event_ind == N_events:
                break

            # Yield after checking there are no more events
            yield current_state

            cur_period += 1        

