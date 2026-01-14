import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from collections import Counter
from collections import defaultdict



################################################## 
kB = 8.617333262145e-5  # eV/K
# -----------------------------
# Geometry utilities
# -----------------------------
def curvature(y, dx):
    N = len(y) - 1
    curv = np.zeros_like(y)
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        curv[i] = (y[ip] - 2*y[i] + y[im]) / dx**2
    curv[N] = curv[0]  # enforce periodic image
    return curv

def arrhenius_rate(G, T, nu0=1e12):
    return nu0 * np.exp(-G / (kB * T))

def rate_diagnostics(events):
    totals = defaultdict(float)
    max_rates = defaultdict(float)
    counts = defaultdict(int)
    for r, etype, *_ in events:
        key = etype.split('_')[0]  # "nuc" or "prop"
        totals[key] += r
        max_rates[key] = max(max_rates[key], r)
        counts[key] += 1
    return totals, max_rates, counts

# -----------------------------
# Event construction
# -----------------------------
def build_events(y, Gn, Gp, T, stress, b, Gamma_n, Gamma_p, dx, Omega_n, Omega_p):
    events = []
    N = len(y) - 1
    curv = curvature(y, dx)

    # ---- Nucleation: single combined event ----
    for i in range(N):
        ip = (i + 1) % N
        jump = y[ip] - y[i]

        # Skip if step already exists
        if jump != 0:
            continue

        # Local nucleation barrier
        Gs = Gn[i] + Gamma_n * np.abs(curv[i])
        rate = arrhenius_rate(Gs - stress * Omega_n, T)
        # single nucleation event affecting both sites
        events.append((rate, "nuc", i, {"jump": jump, "curv": curv[i], "Gs": Gs}))

    # ---- Propagation: left & right ----
    for j in range(N):
        jp = (j + 1) % N
        if y[j] != y[jp]:  # step exists
            Gs = Gp[j] + Gamma_p * np.abs(curv[j])
            rate_r = arrhenius_rate(Gs - stress * Omega_p, T)
            rate_l = arrhenius_rate(Gs + stress * Omega_p, T)
            events.append((rate_r, "prop_r", j, None))
            events.append((rate_l, "prop_l", j, None))

    return events


# -----------------------------
# Event execution
# -----------------------------
def execute_event(event, y, b, histories):
    rate, etype, j, meta = event
    nuc_history, prop_history_r, prop_history_l = histories
    N = len(y) - 1
    jp = (j + 1) % N

    if etype == "nuc":
        y[j] += b
        y[jp] += b
        nuc_history[j] += 1

    elif etype == "prop_r":
        y[jp] += b
        prop_history_r[j] += 1

    elif etype == "prop_l":
        y[j] += b
        prop_history_l[j] += 1

    # enforce periodic image
    y[N] = y[0]


# -----------------------------
# Main KMC simulation without masks
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
    steps=300
):
    x = np.arange(N + 1) * dx
    y = np.zeros(N + 1)

    Gn = np.random.normal(mu_n, sig_n, N)
    Gp = np.random.normal(mu_p, sig_p, N)

    y_snapshots = []
    event_times = []

    time = 0.0

    # History counters
    nuc_history = np.zeros(N, dtype=int)
    prop_history_r = np.zeros(N, dtype=int)
    prop_history_l = np.zeros(N, dtype=int)

    for step in range(steps):
        events = build_events(y, Gn, Gp, T, stress, b, Gamma_n, Gamma_p, dx, Omega_n, Omega_p)

        if not events:
            print("No events available — terminating.")
            break

        rates = np.array([ev[0] for ev in events])
        Rtot = rates.sum()

        # Choose event
        probs = rates / Rtot
        idx = np.random.choice(len(events), p=probs)
        event = events[idx]

        # Advance time
        time += np.random.exponential(1.0 / Rtot)

        # Execute
        execute_event(event, y, b,
                      histories=(nuc_history, prop_history_r, prop_history_l))

        y_snapshots.append(y.copy())
        event_times.append(time)

        # Diagnostics every 10 steps
        if step % 10 == 0:
            totals, max_rates, counts = rate_diagnostics(events)
            print(f"\n--- Step {step} diagnostics ---")
            print("Event counts:", dict(counts))
            print("Total rates:", {k: f"{v:.2e}" for k, v in totals.items()})
            print("Max single-event rates:", {k: f"{v:.2e}" for k, v in max_rates.items()})
            repeaters = {i: c for i, c in enumerate(nuc_history) if c > 1}
            print("Repeated nucleation sites:", repeaters)
            print("Forward nucleations per site:", np.nonzero(nuc_history))
            print("Right propagation per site:", np.nonzero(prop_history_r))
            print("Left propagation per site:", np.nonzero(prop_history_l))

    return {
        "x": x,
        "y_snapshots": np.array(y_snapshots),
        "event_times": np.array(event_times),
        "Gn": Gn,
        "Gp": Gp
    }




