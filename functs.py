import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.animation as animation

# -----------------------------
# Utility: curvature
# -----------------------------
def curvature(y, dx=1.0):
    y = np.asarray(y)
    N = len(y)
    curv = np.zeros_like(y)
    for i in range(1, N-1):
        curv[i] = y[i+1] - 2*y[i] + y[i-1]
    curv[0] = curv[1]
    curv[-1] = curv[-2]
    return curv / dx**2

# -----------------------------
# Safe exponential
# -----------------------------
MAX_EXP = 700
MIN_RATE = 1e-50  # prevents zero rates
def safe_exp(x):
    return np.exp(np.clip(x, -MAX_EXP, MAX_EXP))

def safe_rate(G, kB, T):
    # Clamp exponent
    exponent = -G / (kB*T)
    exponent = np.clip(exponent, -MAX_EXP, MAX_EXP)
    return max(np.exp(exponent), MIN_RATE)

# -----------------------------
# Build event list
# -----------------------------
def build_events(y, Gn, Gp, T, stress, b, Gamma_n, Gamma_p, dx=1.0,
                 Omega_n=0.5, Omega_p=0.5, pbc=True):
    kB = 8.617e-5
    N = len(y)
    events = []

    curv = curvature(y, dx)
    Gn_eff = np.clip(Gn + Gamma_n * curv, 1e-4, None)

    # --- Nucleation at all nodes, bidirectional ---
    for node in range(N):
        rate_fwd = safe_rate(Gn_eff[node] - stress*Omega_n, kB, T)
        events.append((rate_fwd, 'nucleation_fwd', node))

        rate_bwd = safe_rate(Gn_eff[node] + stress*Omega_n, kB, T)
        events.append((rate_bwd, 'nucleation_bwd', node))

    # --- Propagation: all nodes with a kink (y != 0) ---
    kink_nodes = np.where(y != 0)[0]
    for node in kink_nodes:
        # Forward propagation
        next_idx = (node + 1) % N if pbc else node + 1
        if next_idx < N:
            G_saddle_fwd = Gp[min(node, len(Gp)-1)] + Gamma_p * curv[node] - stress * Omega_p
            rate_fwd = safe_rate(G_saddle_fwd, kB, T)
            events.append((rate_fwd, 'propagate_fwd', node))

        # Backward propagation
        prev_idx = (node - 1) % N if pbc else node - 1
        if prev_idx >= 0:
            G_saddle_bwd = Gp[min(prev_idx, len(Gp)-1)] + Gamma_p * curv[node] + stress * Omega_p
            rate_bwd = safe_rate(G_saddle_bwd, kB, T)
            events.append((rate_bwd, 'propagate_bwd', node))

    return events

# -----------------------------
# Execute event
# -----------------------------
def execute_event(event, y, b, pbc=True):
    typ = event[1]
    node = event[2]
    N = len(y)

    if typ == 'nucleation_fwd':
        y[node] += b
    elif typ == 'nucleation_bwd':
        y[node] -= b
    elif typ == 'propagate_fwd':
        next_idx = (node + 1) % N if pbc else node + 1
        if next_idx < N:
            y[next_idx] += b
    elif typ == 'propagate_bwd':
        prev_idx = (node - 1) % N if pbc else node - 1
        if prev_idx >= 0:
            y[prev_idx] += b

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
    Omega_n=0.5,
    Omega_p=0.5,
    steps=300,
    pbc=True
):
    x = np.arange(N) * dx
    y = np.zeros(N)

    # Nucleation and propagation barriers
    Gn = np.random.normal(mu_n, sig_n, N)
    Gp = np.random.normal(mu_p, sig_p, N-1)

    y_snapshots = []
    event_times = []
    time = 0.0

    for step in range(steps):
        events = build_events(y, Gn, Gp, T, stress, b, Gamma_n, Gamma_p, dx,
                              Omega_n, Omega_p, pbc)

        rates = np.array([ev[0] for ev in events])
        Rtot = rates.sum()
        if Rtot == 0:
            print("No events available — terminating.")
            break

        # Normalize probabilities safely
        probs = rates / Rtot
        probs /= probs.sum()  # ensures sum=1

        # Choose event
        idx = np.random.choice(len(events), p=probs)
        event = events[idx]

        # Advance time
        dt = np.random.exponential(1.0 / Rtot)
        time += dt

        # Execute
        execute_event(event, y, b, pbc)

        # Save snapshot
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