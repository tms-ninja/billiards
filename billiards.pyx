# distutils: language = c++
from cython_header cimport *

import numpy as np
cimport numpy as np

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


# class PyEvent

cdef class PyEvent():
    """Represents a collision event of a disc"""

    # We hold onto the PySim instance and use an index to avoid memory
    # management issues that may arise when using a pointer
    cdef PySim _sim
    cdef size_t _e_ind

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
    def second_ind(self):
        """
        Index of secondary object (disc or wall)
        
        Parameters
        ----------
        None.

        Returns
        -------
        int
            Returns the index of the secondary object
            
        """

        return self._sim.s.events[self._e_ind].second_ind

    @property
    def disc_wall_col(self):
        """
        Indicates whether the collision involved a wall or another disc
        
        Parameters
        ----------
        None.

        Returns
        -------
        bool
            Returns True if the secondary object was a wall, False if was 
            anopther disc.
            
        """

        return self._sim.s.events[self._e_ind].disc_wall_col

    @property
    def pos(self):
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
            n[i] = self._sim.s.events[self._e_ind].pos[i]

        return n

    @property
    def new_v(self):
        """
        New velocity of disc after collision
        
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
            n[i] = self._sim.s.events[self._e_ind].new_v[i]

        return n

# class PySim

cdef class PySim():
    """
    Represents a simulation of discs, not intended to be initialized directly
    """

    cdef Sim* s
    cdef list _events

    def __init__(self, bottom_left, top_right):
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

        self.s = new Sim(v_bottom_left, v_top_right)


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

    def setup(self):
        """
        Sets up the simulation internally so it is ready to run. Should be 
        called before calling PySim.advance()
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        self.s.setup()

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

    def add_disc(self, r, v, double m, double R):
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

        Returns
        -------
        None.

        """

        cdef Vec2D v_r = Vec2D(r[0], r[1])
        cdef Vec2D v_v = Vec2D(v[0], v[1])
        cdef Disc d = Disc(v_r, v_v, m, R)

        self.s.initial_state.push_back(d)

    def add_random_discs(self, bottom_left, top_right, N_discs, v, m, R):
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
        v : float or numpy.ndarray
            The speed added discs will have. If a numpy array is passed, it 
            should have shape (N_discs,).
        m : float or numpy.ndarray
            The mass added discs will have. If a numpy array is passed, it 
            should have shape (N_discs,).
        R : float or numpy.ndarray
            The radius added discs will have. If a numpy array is passed, it 
            should have shape (N_discs,).

        Returns
        -------
        None.

        Raises
        ------
        RuntimeError
            Raised if it failed to place a disc after 10 attempts. No new discs
            are added to the simulation if this is raised.
        
        """

        current_state = self.current_state

        N_current_state = current_state['m'].shape[0]

        mass = np.empty(N_current_state + N_discs, dtype=np.float64)

        mass[:N_current_state] = current_state['m']
        mass[N_current_state:] = m

        radius = np.empty(N_current_state + N_discs, dtype=np.float64)

        radius[:N_current_state] = current_state['R']
        radius[N_current_state:] = R

        velocity = np.empty((N_current_state + N_discs, 2), dtype=np.float64)
        angle = 2*np.pi*np.random.random(N_discs)

        velocity[:N_current_state] = current_state['v']
        velocity[N_current_state:, 0] = v*np.cos(angle)
        velocity[N_current_state:, 1] = v*np.sin(angle)

        # Disc centres must be in a smaller box so they don't intersect the
        # walls
        bottom_left = bottom_left + R
        top_right = top_right - R

        pos = np.empty((N_current_state + N_discs, 2), dtype=np.float64)
        pos[:N_current_state] = current_state['r']

        for d_ind in range(N_current_state, N_current_state + N_discs):
            attempt = 10

            while attempt > 0:
                d_pos = bottom_left + (top_right - bottom_left) * np.random.random(2)

                disc_is_colliding = False

                # Test for collisions
                for i in range(0, d_ind):
                    if np.linalg.norm(d_pos - pos[i]) < radius[d_ind] + radius[i]:
                        disc_is_colliding = True
                        break
                else:
                    pos[d_ind] = d_pos
                    break  # New disc has no collisions

                # Disc does have a collision, try again
                attempt -= 1
            else:
                raise RuntimeError(f"Unable to place disc {d_ind - N_current_state} after 10 attempts.")

        # Now add the discs to the simulation
        for ind in range(N_current_state, N_current_state + N_discs):
            self.add_disc(pos[ind], velocity[ind], mass[ind], radius[ind]) 


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
            - 'm' - mass
            - 'R' - radius
        """

        state_dict = {}

        state_dict['r'] = _get_state_pos(self.s.initial_state)
        state_dict['v'] = _get_state_v(self.s.initial_state)
        state_dict['m'] = _get_state_m(self.s.initial_state)
        state_dict['R'] = _get_state_R(self.s.initial_state)

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
            - 'm' - mass
            - 'R' - radius
        """

        state_dict = {}

        state_dict['r'] = np.empty((self.s.initial_state.size(), 2), dtype=np.float64)
        state_dict['v'] = np.empty((self.s.initial_state.size(), 2), dtype=np.float64)

        # Assume mass and radii of discs don't change during simulation
        state_dict['m'] = _get_state_m(self.s.initial_state)
        state_dict['R'] = _get_state_R(self.s.initial_state)

        cdef Event cur_disc


        for d_ind in range(0, self.s.initial_state.size()):
            cur_disc = self.s.events_vec[d_ind][self.s.old_vec[d_ind]]

            for p_ind in range(0, 2):
                state_dict['r'][d_ind, p_ind] = cur_disc.pos[p_ind]
                state_dict['v'][d_ind, p_ind] = cur_disc.new_v[p_ind]

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
            - 'm' - mass
            - 'R' - radius
        
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
            current_state['r'] += dt*current_state['v']

            # Update the colliding particle accordingly
            for ind in range(2):
                current_state['v'][cur_disc][ind] = self.s.events[current_state_ind].new_v[ind]

            current_t = self.s.events[current_state_ind].t

            yield current_state

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
            - 'm' - mass
            - 'R' - radius
        
        """

        current_state = self.initial_state

        yield current_state

        cdef size_t N_events = self.s.events.size()
        cdef size_t cur_event_ind = 0
        cdef size_t cur_disc
        cdef double current_t = 0.0
        cdef double time_step

        # Start time for the current period
        cdef int cur_period = 1

        # The inner loop corresponds to advancing one event at a time, the 
        # outer loop corresponds to advancing by dt
        while True:

            while cur_event_ind < N_events:
                # Next event is in the current period
                if self.s.events[cur_event_ind].t <= cur_period*dt:
                    # advance to next event
                    time_step = self.s.events[cur_event_ind].t - current_t
                    current_t = self.s.events[cur_event_ind].t

                    current_state['r'] += time_step*current_state['v']

                    cur_disc = self.s.events[cur_event_ind].ind

                    # Update the colliding particle accordingly
                    for ind in range(2):
                        current_state['v'][cur_disc][ind] = self.s.events[cur_event_ind].new_v[ind]
                else:
                    # advance to end of time interval and break
                    time_step = cur_period*dt - current_t
                    current_t = cur_period*dt

                    current_state['r'] += time_step*current_state['v']

                    break

                cur_event_ind += 1
            
            # No more events to process
            if cur_event_ind == N_events:
                break

            # Yield after checking there are no more events
            yield current_state

            cur_period += 1        

