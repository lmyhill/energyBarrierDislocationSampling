import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

################################################## 

import numpy as np

kB = 8.617333262145e-5  # eV/K


# -----------------------------
# Geometry utilities
# -----------------------------
def curvature(y, dx):
    curv = np.zeros_like(y)
    N = len(y) - 1  # last node is periodic image

    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        curv[i] = (y[ip] - 2*y[i] + y[im]) / dx**2

    # enforce periodic image
    curv[N] = curv[0]
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
    for i in range(N):
        G_eff = Gn[i] + Gamma_n * np.abs(curv[i])
        rate = arrhenius_rate(G_eff, T)
        events.append((rate, "nucleate", i))


    # ---- Propagation (front advance) ----
    for j in range(N):
        jp = (j + 1) % N  # periodic neighbor

        # Right-moving front
        if y[j] > y[jp]:
            G_eff = (
                Gp[j]
                + Gamma_p * np.abs(curv[j])
                - stress * Omega_p
            )
            rate = arrhenius_rate(G_eff, T)
            events.append((rate, "prop_r", j))

        # Left-moving front
        if y[jp] > y[j]:
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
    N = len(y) - 1

    if etype == "nucleate":
        y[j] += b

    elif etype == "prop_r":
        y[(j + 1) % N] += b

    elif etype == "prop_l":
        y[j] += b

    # enforce periodic image
    y[N] = y[0]





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



