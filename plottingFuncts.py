import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.animation as animation




# -----------------------------
# Visualization
# -----------------------------
def plot_line_trajectory_colored(x, y_snapshots):
    segments = []
    colors = []

    for k in range(len(y_snapshots)):
        yk = y_snapshots[k]

        # truncate x to match y
        x_plot = x[:len(yk)]
        y_plot = yk

        segments.append(np.column_stack((x_plot, y_plot)))
        colors.append(k)

    fig, ax = plt.subplots(figsize=(7, 4))

    lc = LineCollection(segments, cmap="viridis", linewidths=1.2)
    lc.set_array(np.array(colors))

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title("Dislocation Line Trajectory")

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Event index")
    fig.tight_layout()
    return fig, ax




def create_animation(x, y_snapshots, filename='line_trajectory.gif', interval=100):

    fig, ax = plt.subplots(figsize=(7,4))
    line, = ax.plot([], [], lw=2)

    # Determine limits safely
    y_min = np.min(y_snapshots)
    y_max = np.max(y_snapshots)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title("Dislocation Line Trajectory")

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        yk = y_snapshots[i]
        N = len(yk)

        # truncate x to match y if needed
        xk = x[:N]

        line.set_data(xk, yk)
        return (line,)

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(y_snapshots), interval=interval, blit=True
    )

    ani.save(filename, writer='pillow')
    plt.close()


# -----------------------------
# More output functions
# -----------------------------
def return_velocity(event_times, line_length):
    total_time = event_times[-1] - event_times[0]
    total_displacement = line_length * (len(event_times) - 1)
    return total_displacement / total_time if total_time > 0 else 0.0

def return_num_kinks(y_snapshots):
    num_kinks = []
    for y in y_snapshots:
        dy = np.diff(y)
        kinks = np.sum(dy > 0)
        num_kinks.append(kinks)
    return np.array(num_kinks)

def plot_num_kinks(x, y_snapshots):
    num_kinks = return_num_kinks(y_snapshots)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, num_kinks, marker='o')
    ax.set_xlabel("Event index")
    ax.set_ylabel("Number of Kinks")
    ax.set_title("Number of Kinks Over Time")
    fig.tight_layout()

    return fig, ax

def plot_velocity(event_times, line_length):
    velocities = []
    for i in range(1, len(event_times)):
        dt = event_times[i] - event_times[i-1]
        v = line_length / dt if dt > 0 else 0.0
        velocities.append(v)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(1, len(event_times)), velocities, marker='o')
    ax.set_xlabel("Event index")
    ax.set_ylabel("Velocity")
    ax.set_title("Velocity Over Time")
    fig.tight_layout()

    return fig, ax
# -----------------------------
