# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool



cdef extern from "Vec2D.cpp":
    pass

cdef extern from "Vec2D.h":
    cdef cppclass Vec2D:
        Vec2D() except+
        Vec2D(double, double) except+
        double& operator[](size_t)


cdef extern from "Disc.cpp":
    pass

cdef extern from "Disc.h":
    cdef cppclass Disc:
        Disc() except+
        Disc(Vec2D&, Vec2D&, double,double) except+
        Vec2D r
        Vec2D v
        double m
        double R


cdef extern from "Wall.cpp":
    pass

cdef extern from "Wall.h":
    cdef cppclass Wall:
        Wall() except+
        Wall(Vec2D&, Vec2D&) except+
        Vec2D start
        Vec2D end
        Vec2D tangent


cdef extern from "Event.cpp":
    pass

cdef extern from "Event.h":
    cdef cppclass Event:
        Event() except+
        Event(double, size_t, size_t, bool, Vec2D&, Vec2D&) except+
        Vec2D pos
        Vec2D new_v
        double t
        size_t ind
        size_t second_ind
        bool disc_wall_col

# Declare std::array<Event, 2>
cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass Array_Event "std::array<Event, 2>":
        Array_Event() except+
        Event& operator[](size_t)


cdef extern from "Sim.cpp":
    pass
    
cdef extern from "Sim.h":
    cdef cppclass Sim:
        Sim() except+
        vector[Disc] initial_state

        vector[size_t] new_vec
        vector[size_t] old_vec

        vector[Array_Event] events_vec  # Used for book keeping, events_vec[i, old_vec[i]] contains pseudo current_state

        vector[Wall] walls
        vector[Event] events  # events that occured during the simulation
        double current_time

        void advance(size_t, double, bool)
        void setup()
    
