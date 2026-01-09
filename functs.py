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
# def simulate_line(
#     N=50,
#     b=1.0,
#     dx=1.0,
#     T=300,
#     stress=0.1,
#     mu_n=0.2,
#     sig_n=0.1,
#     mu_p=0.05,
#     sig_p=0.05,
#     Gamma_n=0.5,
#     Gamma_p=0.1,
#     steps=300
# ):
#     x = np.arange(N+1) * dx

#     Gn = sample_barriers(N+1, mu_n, sig_n)
#     Gp = sample_barriers(N,   mu_p, sig_p)

#     y = np.zeros(N+1)

#     y_snapshots = []
#     event_times = []
#     time = 0.0

#     for step in range(steps):
#         # --- Nucleation ---
#         Gn_eff = effective_nucleation_barrier(
#             Gn, y, Gamma_n, dx
#         )
#         lambda_n = arrhenius_rate(Gn_eff, T)

#         probs = lambda_n / np.sum(lambda_n)
#         node = np.random.choice(N+1, p=probs)

#         tn = np.random.exponential(1.0 / lambda_n[node])
#         time += tn

#         # --- Propagation ---
#         propagate_kink(
#             y, node, Gp, T, stress, b,
#             Gamma_p, dx
#         )

#         y_snapshots.append(y.copy())
#         event_times.append(time)
        
#         # Print a progress update
#         if step % 10 == 0:
#             print(f"Completed step {step+1} / {steps}")
        
#     return {
#         "x": x,
#         "y_snapshots": np.array(y_snapshots),
#         "event_times": np.array(event_times),
#         "Gn": Gn,
#         "Gp": Gp
#     }


# -----------------------------
# Propagation with line tension
# -----------------------------
# def propagate_kink(
#     y,
#     node,
#     Gp,
#     T,
#     stress,
#     b,
#     Gamma_p,
#     dx=1.0,
#     Omega_p=0.5
# ):
#     kB=8.617333262145e-5  # eV/K
#     N = len(y) - 1
#     curv = np.abs(curvature(y, dx))

#     # reference barrier (typical pinning)
#     Gp_ref = np.median(Gp)

#     # Right
#     for j in range(node, N):
#         G_eff = (
#             Gp[j]
#             + Gamma_p * curv[j]
#             - stress * Omega_p
#         )

#         delta = max(G_eff - Gp_ref, 0.0)
#         p_stop = 1.0 - np.exp(-delta / (kB * T))

#         if np.random.rand() < p_stop:
#             break

#         y[j+1] += b

#     # Left
#     for j in range(node-1, -1, -1):
#         G_eff = (
#             Gp[j]
#             + Gamma_p * curv[j]
#             - stress * Omega_p
#         )

#         delta = max(G_eff - Gp_ref, 0.0)
#         p_stop = 1.0 - np.exp(-delta / (kB * T))

#         if np.random.rand() < p_stop:
#             break

#         y[j] += b

# # -----------------------------
# # Nucleation barrier with line tension
# # -----------------------------
# def effective_nucleation_barrier(Gn, y, Gamma_n, dx=1.0):
#     return np.clip(
#         Gn + Gamma_n * np.abs(curvature(y, dx)),
#         1e-4,
#         None
#     )


# -----------------------------
# Utilities
# -----------------------------
# def arrhenius_rate(barrier, T, tau0=1e-12):
#     kB=8.617333262145e-5  # eV/K
#     return (1.0 / tau0) * np.exp(-barrier / (kB * T))

# def sample_barriers(N, mu, sigma):
#     return np.clip(np.random.normal(mu, sigma, size=N), 1e-3, None)

# def curvature(y, dx=1.0):
#     c = np.zeros_like(y)
#     c[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / dx**2
#     return np.abs(c)

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



################################################## 

import numpy as np

kB = 8.617333262145e-5  # eV/K


# -----------------------------
# Geometry utilities
# -----------------------------
def curvature(y, dx):
    curv = np.zeros_like(y)
    curv[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / dx**2
    curv[0] = curv[1]
    curv[-1] = curv[-2]
    return curv


def arrhenius_rate(G, T, nu0=1e12):
    return nu0 * np.exp(-G / (kB * T))


# -----------------------------
# Event construction
# -----------------------------
def build_events(
    y, Gn, Gp, T, stress, b,
    Gamma_n, Gamma_p, dx, Omega_p
):
    events = []
    N = len(y) - 1
    curv = curvature(y, dx)

    # ---- Nucleation ----
    for i in range(N + 1):
        G_eff = Gn[i] + Gamma_n * np.abs(curv[i])
        rate = arrhenius_rate(G_eff, T)
        events.append((rate, "nucleate", i))

    # ---- Propagation (front advance) ----
    for j in range(N):
        # Right-moving front
        if y[j] > y[j+1]:
            G_eff = (
                Gp[j]
                + Gamma_p * np.abs(curv[j])
                - stress * Omega_p
            )
            rate = arrhenius_rate(G_eff, T)
            events.append((rate, "prop_r", j))

        # Left-moving front
        if y[j+1] > y[j]:
            G_eff = (
                Gp[j]
                + Gamma_p * np.abs(curv[j])
                + stress * Omega_p
            )
            rate = arrhenius_rate(G_eff, T)
            events.append((rate, "prop_l", j))
    return events




# -----------------------------
# Event execution
# -----------------------------
def execute_event(event, y, b):
    _, etype, j = event

    if etype == "nucleate":
        y[j] += b

    elif etype == "prop_r":
        y[j+1] += b

    elif etype == "prop_l":
        y[j] += b




# -----------------------------
# Main KMC simulation
# -----------------------------
def simulate_line_kmc(
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
    Omega_p=0.5,
    steps=300
):
    x = np.arange(N + 1) * dx
    y = np.zeros(N + 1)

    Gn = np.random.normal(mu_n, sig_n, N + 1)
    Gp = np.random.normal(mu_p, sig_p, N)

    y_snapshots = []
    event_times = []

    time = 0.0

    for step in range(steps):
        events = build_events(
            y, Gn, Gp, T, stress, b,
            Gamma_n, Gamma_p, dx, Omega_p
        )

        rates = np.array([ev[0] for ev in events])
        Rtot = rates.sum()

        if Rtot == 0:
            print("No events available — terminating.")
            break

        # Choose event
        probs = rates / Rtot
        idx = np.random.choice(len(events), p=probs)
        event = events[idx]

        # Advance time
        time += np.random.exponential(1.0 / Rtot)

        # Execute
        execute_event(event, y, b)

        y_snapshots.append(y.copy())
        event_times.append(time)

        if step % 10 == 0:
            print(f"Completed step {step+1}/{steps}")

    return {
        "x": x,
        "y_snapshots": np.array(y_snapshots),
        "event_times": np.array(event_times),
        "Gn": Gn,
        "Gp": Gp
    }



