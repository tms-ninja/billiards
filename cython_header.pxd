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


cdef extern from "Sim.cpp":
    pass
    
cdef extern from "Sim.h":
    cdef cppclass Sim:
        Sim() except+
        vector[Disc] current_state
        vector[Disc] initial_state
        vector[Wall] walls
        vector[Event] events
        double current_time

        void advance(size_t, double, bool)
        void setup()
    
