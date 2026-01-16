import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from functs import build_events, kB
import matplotlib.cm as cm

# -----------------------------
# Visualization
# -----------------------------
def plot_line_trajectory_colored(x, y_snapshots):
    segments = []
    colors = []

    for k in range(len(y_snapshots)):
        segments.append(np.column_stack((x, y_snapshots[k])))
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
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(7,4))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y_snapshots), np.max(y_snapshots))
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title("Dislocation Line Trajectory")

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(x, y_snapshots[i])
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

# --- Diagnostic output functions ---


def visualize_animation_with_curvature_annotations(x, y_snapshots, filename='line_trajectory_curvature.gif', interval=100):
    """Visualize animation with curvature annotations at each node."""
    import matplotlib.animation as animation
    
    def compute_curvature(x, y):
        """Compute curvature at each point using finite differences."""
        dx = x[1] - x[0]  # assuming uniform spacing
        curv = np.zeros_like(y)
        N = len(y) - 1  # last node is periodic image

        for i in range(N):
            ip = (i + 1) % N
            im = (i - 1) % N
            curv[i] = (y[ip] - 2*y[i] + y[im]) / dx**2

        # enforce periodic image
        curv[N] = curv[0]
        return curv
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    line, = ax1.plot([], [], lw=2)
    scatter = ax1.scatter([], [], c=[], cmap='coolwarm', s=50, zorder=5)
    
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(np.min(y_snapshots), np.max(y_snapshots))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y(x)")
    ax1.set_title("Dislocation Line Trajectory")
    
    ax2.set_xlim(np.min(x[:-2]), np.max(x[:-2]))
    ax2.set_ylim(0, np.max([np.max(compute_curvature(x, y)) for y in y_snapshots]))
    ax2.set_xlabel("x")
    ax2.set_ylabel("Curvature")
    ax2.set_title("Curvature Distribution")
    
    curvature_line, = ax2.plot([], [], lw=2, color='orange')
    
    def init():
        line.set_data([], [])
        curvature_line.set_data([], [])
        return line, scatter, curvature_line
    
    def animate(i):
        y = y_snapshots[i]
        line.set_data(x, y)
        
        curvature = compute_curvature(x, y)
        scatter.set_offsets(np.c_[x[1:-1], y[1:-1]])
        scatter.set_array(curvature[1:-1])
        
        curvature_line.set_data(x[:-2], curvature[:-2])
        
        return line, scatter, curvature_line
    
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(y_snapshots), interval=interval, blit=True
    )
    
    ani.save(filename, writer='pillow')
    plt.close()
    
    
def visualize_animation_with_deviation_annotations(x, y_snapshots, filename='line_trajectory_deviation.gif', interval=100):
    """Visualize animation with deviation from average annotations at each node."""
    import matplotlib.animation as animation
    
    def compute_deviation(y):
        """Compute deviation of each point from the average."""
        avg = np.mean(y[:-1])  # excluding periodic image
        return y[:-1] - avg
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    line, = ax1.plot([], [], lw=2)
    scatter = ax1.scatter([], [], c=[], cmap='coolwarm', s=50, zorder=5)
    
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(np.min(y_snapshots), np.max(y_snapshots))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y(x)")
    ax1.set_title("Dislocation Line Trajectory")
    
    ax2.set_xlim(np.min(x[:-1]), np.max(x[:-1]))
    dev_max = np.max([np.max(np.abs(compute_deviation(y))) for y in y_snapshots])
    ax2.set_ylim(-dev_max, dev_max)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Deviation from Average")
    ax2.set_title("Point Deviation Distribution")
    
    deviation_scatter = ax2.scatter([], [], c=[], cmap='coolwarm', s=50, zorder=5)
    
    def init():
        line.set_data([], [])
        return line, scatter, deviation_scatter
    
    def animate(i):
        y = y_snapshots[i]
        line.set_data(x, y)
        
        deviation = compute_deviation(y)
        scatter.set_offsets(np.c_[x[:-1], y[:-1]])
        scatter.set_array(deviation)
        
        deviation_scatter.set_offsets(np.c_[x[:-1], deviation])
        deviation_scatter.set_array(deviation)
        
        return line, scatter, deviation_scatter
    
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(y_snapshots), interval=interval, blit=True
    )
    
    ani.save(filename, writer='pillow')
    plt.close()
    
def visualize_animation_with_rateCalculation_annotations(x, y_snapshots, Gn, Gp, T, stress, b, Gamma_n, Gamma_p, dx, Omega_n, Omega_p, filename='line_trajectory_rateCalculation.gif', interval=100):
    """Visualize animation with rate calculation annotations at each node."""
    import matplotlib.animation as animation
    
    def compute_rate_calculation(y):
        """Compute rate values at each node from all possible events."""
        events = build_events(y, Gn, Gp, T, stress, b, Gamma_n, Gamma_p, dx, Omega_n, Omega_p)
        N = len(y) - 1
        rate_per_node = np.zeros(N)
        
        # Aggregate rates by node
        for rate, event_type, node_idx in events:
            rate_per_node[node_idx] += rate
        
        return rate_per_node
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    line, = ax1.plot([], [], lw=2)
    scatter = ax1.scatter([], [], c=[], cmap='coolwarm', s=50, zorder=5)
    
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(np.min(y_snapshots), np.max(y_snapshots))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y(x)")
    ax1.set_title("Dislocation Line Trajectory")
    
    ax2.set_xlim(np.min(x[:-1]), np.max(x[:-1]))
    rate_max = np.max([np.max(compute_rate_calculation(y)) for y in y_snapshots])
    ax2.set_ylim(0, rate_max if rate_max > 0 else 1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Total Event Rate")
    ax2.set_title("Event Rate Distribution Per Node")
    
    rate_scatter = ax2.scatter([], [], c=[], cmap='coolwarm', s=50, zorder=5)
    
    def init():
        line.set_data([], [])
        return line, scatter, rate_scatter
    
    def animate(i):
        y = y_snapshots[i]
        line.set_data(x, y)
        
        rate_values = compute_rate_calculation(y)
        scatter.set_offsets(np.c_[x[:-1], y[:-1]])
        scatter.set_array(rate_values)
        
        rate_scatter.set_offsets(np.c_[x[:-1], rate_values])
        rate_scatter.set_array(rate_values)
        
        return line, scatter, rate_scatter
    
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(y_snapshots), interval=interval, blit=True
    )
    
    ani.save(filename, writer='pillow')
    plt.close()