import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
# Main simulation
# -----------------------------
def simulate_line(
    N=50,
    b=1.0,
    dx=1.0,
    T=300,
    stress=0.1,
    mu_n=0.2,
    sig_n=0.1,
    mu_p=0.05,
    sig_p=0.05,
    Gamma_n=0.5,
    Gamma_p=0.1,
    steps=300
):
    x = np.arange(N+1) * dx

    Gn = sample_barriers(N+1, mu_n, sig_n)
    Gp = sample_barriers(N,   mu_p, sig_p)

    y = np.zeros(N+1)

    y_snapshots = []
    event_times = []
    time = 0.0

    for step in range(steps):
        # --- Nucleation ---
        Gn_eff = effective_nucleation_barrier(
            Gn, y, Gamma_n, dx
        )
        lambda_n = arrhenius_rate(Gn_eff, T)

        probs = lambda_n / np.sum(lambda_n)
        node = np.random.choice(N+1, p=probs)

        tn = np.random.exponential(1.0 / lambda_n[node])
        time += tn

        # --- Propagation ---
        propagate_kink(
            y, node, Gp, T, stress, b,
            Gamma_p, dx
        )

        y_snapshots.append(y.copy())
        event_times.append(time)

    return {
        "x": x,
        "y_snapshots": np.array(y_snapshots),
        "event_times": np.array(event_times),
        "Gn": Gn,
        "Gp": Gp
    }


# -----------------------------
# Propagation with line tension
# -----------------------------
def propagate_kink(
    y,
    node,
    Gp,
    T,
    stress,
    b,
    Gamma_p,
    dx=1.0,
    Omega_p=0.5
):
    kB=8.617333262145e-5  # eV/K
    N = len(y) - 1
    curv = curvature(y, dx)

    # reference barrier (typical pinning)
    Gp_ref = np.median(Gp)

    # Right
    for j in range(node, N):
        G_eff = (
            Gp[j]
            + Gamma_p * curv[j]
            - stress * Omega_p
        )

        delta = max(G_eff - Gp_ref, 0.0)
        p_stop = 1.0 - np.exp(-delta / (kB * T))

        if np.random.rand() < p_stop:
            break

        y[j+1] += b

    # Left
    for j in range(node-1, -1, -1):
        G_eff = (
            Gp[j]
            + Gamma_p * curv[j]
            - stress * Omega_p
        )

        delta = max(G_eff - Gp_ref, 0.0)
        p_stop = 1.0 - np.exp(-delta / (kB * T))

        if np.random.rand() < p_stop:
            break

        y[j] += b

# -----------------------------
# Nucleation barrier with line tension
# -----------------------------
def effective_nucleation_barrier(Gn, y, Gamma_n, dx=1.0):
    return np.clip(
        Gn + Gamma_n * curvature(y, dx),
        1e-4,
        None
    )


# -----------------------------
# Utilities
# -----------------------------
def arrhenius_rate(barrier, T, tau0=1e-12):
    kB=8.617333262145e-5  # eV/K
    return (1.0 / tau0) * np.exp(-barrier / (kB * T))

def sample_barriers(N, mu, sigma):
    return np.clip(np.random.normal(mu, sigma, size=N), 1e-3, None)

def curvature(y, dx=1.0):
    c = np.zeros_like(y)
    c[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / dx**2
    return np.abs(c)