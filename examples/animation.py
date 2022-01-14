# Example animation
from pprint import pprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import billiards as bl

np.set_printoptions(precision=17)
np.random.seed(30)

# First setup the simulation
walls = [
    {'start': [ 0.0,    0.0 ], 'end': [  0.0,  20.0 ]},
    {'start': [ 0.0,   20.0 ], 'end': [ 20.0,  20.0 ]},
    {'start': [ 20.0,  20.0 ], 'end': [ 20.0,   0.0 ]},
    {'start': [ 20.0,   0.0 ], 'end': [ 0.0,    0.0 ]},
]

box_bottom_left = [0.0, 0.0]
box_top_right = [20.0, 20.0]

s = bl.PySim(box_bottom_left, box_top_right)

s.add_random_discs(np.array(box_bottom_left), np.array(box_top_right), 20, 5.0, 1.0, 1.0)


# Set up the simulation ready to be run
s.setup()

# Perform some collision
s.advance(400, 100.0, True)

# Now plot the animation

def total_KE(m, v):
    ke = m * np.linalg.norm(v, axis=1)**2

    return np.sum(ke) / 2.0

def check_overlap(pos, R):

    for i in range(pos.shape[0]):

        # Test outside of walls
        if pos[i, 0] < box_bottom_left[0] or pos[i, 0] > box_top_right[0] or pos[i, 1] < box_bottom_left[1] or pos[i, 1] > box_top_right[1]:
            raise RuntimeError(f"Disc {i} is out of bounds at position {pos[i]}")

        # Test with other discs
        for j in range(i+1, pos.shape[0]):
            if np.linalg.norm(pos[i] - pos[j]) - R[i] - R[j] < -1e-14:
                raise RuntimeError(f"Discs {i} and {j} are overlapping by {np.linalg.norm(pos[i] - pos[j]) - R[i] - R[j]}")

def create_animation(sim, fps=30, slowdown=1):
    """Creates an animation using matplotlib"""
    tmax = int(sim.events[-1].t*fps)

    time_replay = sim.replay_by_time(1/(fps*slowdown))

    def animate(i):
        """animation function"""
        c_state = next(time_replay)

        check_overlap(c_state['r'], c_state['R'])

        print(f"Total KE: {total_KE(c_state['m'], c_state['v'])}\tTime: {i * 1.0 / (fps*slowdown)}")

        d_plot.set_offsets(c_state['r'])

        return [d_plot]

    c_state = next(time_replay)
    pos = c_state['r']


    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    xlim = [0.0, 20.0]
    ylim = [0.0, 20.0]
    plt.xlim(xlim)
    plt.ylim(ylim)

    # plot walls
    for w in walls:
        s, e = w['start'], w['end']
        plt.plot([s[0], e[0]], [s[1], e[1]], color='k')

    # Size of discs
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in_points = bbox.width * 72

    s = 4 * ( width_in_points * c_state['R']/(xlim[1] - xlim[0])) ** 2

    d_plot = plt.scatter(pos[:, 0], pos[:, 1], s=s, alpha=0.5)

    ax.set_aspect('equal')

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames = int(tmax * fps) - 3,
        interval = 1000.0 / fps, # in ms\n",
        repeat = False,
        blit=True
        )

    plt.show()

    return anim


create_animation(s, fps=30, slowdown=1)

