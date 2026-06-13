import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plot_utils import ensure_output_dir

# ==================================================
# Plant parameters (true -- unknown to the controller)
# ==================================================
a1 = 2.0
a2 = 3.0
a3 = 1.0
b1 = 0.05
b2 = 0.1
b3 = 1.5
c = 1.0

theta_true = np.array([a1, a2, a3, b1, b2, b3])

# ==================================================
# Reference model parameters
# ==================================================
zeta = 0.7
wn = 2.0
p3 = 10.0 * zeta * wn

alpha2 = 2.0 * zeta * wn + p3
alpha1 = wn**2 + 2.0 * zeta * wn * p3
alpha0 = wn**2 * p3

# ==================================================
# Controller parameters
# ==================================================
lambda1 = 4.0
lambda2 = 4.0
kz = 5.0
Gamma = 10.0 * np.eye(6)

# ==================================================
# Sampling period
# ==================================================
Ts = 0.01

# ==================================================
# Reference input (constant step setpoint, pi/4 rad)
# ==================================================
def r_cmd(t):
    return np.pi / 4

# ==================================================
# Continuous reference model dynamics
# ==================================================
def reference_model_derivative(t, xm):
    ym, ym_dot, ym_ddot = xm
    ym_3 = (
        -alpha2 * ym_ddot
        -alpha1 * ym_dot
        -alpha0 * ym
        + alpha0 * r_cmd(t)
    )
    return np.array([ym_dot, ym_ddot, ym_3])

# ==================================================
# Continuous plant dynamics (u held by ZOH between samples)
# ==================================================
def plant_derivative(t, xp, u_zoh):
    y, y_dot, y_ddot = xp
    y_3 = (
        -a1 * y_ddot
        -a2 * y_dot
        -a3 * y
        -b1 * y_ddot**3
        -b2 * y_dot**3
        -b3 * np.sin(y)
        + c * u_zoh
    )
    return np.array([y_dot, y_ddot, y_3])

# ==================================================
# Sampled-data simulation
# ==================================================
t_start = 0.0
t_end = 40.0
num_steps = int((t_end - t_start) / Ts)

# Continuous plant and reference model states (sampled at t_k)
xp = np.zeros(3)
xm = np.zeros(3)

# Discrete adaptive parameter estimates at sample k
theta_hat_k = np.zeros(6)

# Error history: e_km1 = e[k-1], e_km2 = e[k-2]
e_km1 = 0.0
e_km2 = 0.0

# Control input applied at sample k (held until next sample by ZOH)
u_k = 0.0

# Logs (one entry per sample)
time_hist = []
r_hist = []
y_hist = []
ym_hist = []
u_hist = []
e_hist = []
z_hist = []
theta_hat_hist = []

for k in range(num_steps + 1):
    t_k = t_start + k * Ts

    # ----- Sample the continuous signals at t_k -----
    y_k, y_dot_k, y_ddot_k = xp
    ym_k, ym_dot_k, ym_ddot_k = xm

    # Reference model third derivative at sample k
    ym_3_k = (
        -alpha2 * ym_ddot_k
        -alpha1 * ym_dot_k
        -alpha0 * ym_k
        + alpha0 * r_cmd(t_k)
    )

    # Tracking error at sample k
    e_k = y_k - ym_k

    # ----- Backward-difference derivative approximations -----
    if k == 0:
        e_dot_k = 0.0
        e_ddot_k = 0.0
    elif k == 1:
        e_dot_k = (e_k - e_km1) / Ts
        e_ddot_k = 0.0
    else:
        e_dot_k = (e_k - e_km1) / Ts
        e_ddot_k = (e_k - 2.0 * e_km1 + e_km2) / Ts**2

    # Filtered error at sample k
    z_k = e_ddot_k + lambda2 * e_dot_k + lambda1 * e_k

    # Auxiliary reference signal at sample k
    yr_3_k = ym_3_k - lambda2 * e_ddot_k - lambda1 * e_dot_k

    # Regressor at sample k
    phi_k = np.array([
        y_ddot_k,
        y_dot_k,
        y_k,
        y_ddot_k**3,
        y_dot_k**3,
        np.sin(y_k),
    ])

    # ----- Discrete adaptive control law -----
    # u_k is held constant by the ZOH until sample k+1
    u_k = (theta_hat_k @ phi_k + yr_3_k - kz * z_k) / c

    # ----- Forward-Euler discretisation of the adaptation law -----
    # theta_hat[k+1] = theta_hat[k] - Ts * Gamma * phi[k] * z[k]
    theta_hat_k = theta_hat_k - Ts * (Gamma @ phi_k) * z_k

    # Log values at sample k
    time_hist.append(t_k)
    r_hist.append(r_cmd(t_k))
    y_hist.append(y_k)
    ym_hist.append(ym_k)
    u_hist.append(u_k)
    e_hist.append(e_k)
    z_hist.append(z_k)
    theta_hat_hist.append(theta_hat_k.copy())

    # Shift error history for next sample
    e_km2 = e_km1
    e_km1 = e_k

    # ----- ZOH integration of plant and reference model over [t_k, t_k + Ts) -----
    # u_k is held constant for the whole interval -- this is what makes it ZOH.
    if k < num_steps:
        t_next = t_k + Ts

        sol_p = solve_ivp(
            lambda t, x: plant_derivative(t, x, u_k),
            (t_k, t_next),
            xp,
            rtol=1e-8,
            atol=1e-10,
        )

        sol_m = solve_ivp(
            reference_model_derivative,
            (t_k, t_next),
            xm,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol_p.success:
            raise RuntimeError("Plant integration failed: " + sol_p.message)
        if not sol_m.success:
            raise RuntimeError("Reference model integration failed: " + sol_m.message)

        xp = sol_p.y[:, -1]
        xm = sol_m.y[:, -1]

# ==================================================
# Convert logs to arrays
# ==================================================
t = np.array(time_hist)
r_vals = np.array(r_hist)
y_vals = np.array(y_hist)
ym_vals = np.array(ym_hist)
u_vals = np.array(u_hist)
e_vals = np.array(e_hist)
z_vals = np.array(z_hist)
theta_hat_vals = np.array(theta_hat_hist).T  # shape (6, n_times) for plot_utils

# ==================================================
# Save figures
# ==================================================
output_dir = "figs/"
ensure_output_dir(output_dir)

# --------------------------------------------------
# Continuous-time simulation (for comparison)
# --------------------------------------------------
def closed_loop_continuous_dynamics(t, X):
    y, y_dot, y_ddot, ym, ym_dot, ym_ddot = X[:6]
    theta_hat = X[6:12]

    ym_3 = (
        -alpha2 * ym_ddot
        -alpha1 * ym_dot
        -alpha0 * ym
        + alpha0 * r_cmd(t)
    )

    e = y - ym
    e_dot = y_dot - ym_dot
    e_ddot = y_ddot - ym_ddot

    z = e_ddot + lambda2 * e_dot + lambda1 * e
    yr_3 = ym_3 - lambda2 * e_ddot - lambda1 * e_dot

    phi = np.array([
        y_ddot, y_dot, y, y_ddot**3, y_dot**3, np.sin(y),
    ])

    u = (theta_hat @ phi + yr_3 - kz * z) / c

    y_3 = (
        -a1 * y_ddot - a2 * y_dot - a3 * y
        - b1 * y_ddot**3 - b2 * y_dot**3 - b3 * np.sin(y)
        + c * u
    )

    theta_hat_dot = -Gamma @ phi * z

    return [
        y_dot, y_ddot, y_3,
        ym_dot, ym_ddot, ym_3,
        theta_hat_dot[0], theta_hat_dot[1], theta_hat_dot[2],
        theta_hat_dot[3], theta_hat_dot[4], theta_hat_dot[5],
    ]


sol_cont = solve_ivp(
    closed_loop_continuous_dynamics,
    (t_start, t_end),
    np.zeros(12),
    t_eval=t,
    rtol=1e-8,
    atol=1e-10,
)
if not sol_cont.success:
    raise RuntimeError("Continuous integration failed: " + sol_cont.message)

y_cont = sol_cont.y[0]
ym_cont = sol_cont.y[3]
e_cont = y_cont - ym_cont
e_dot_cont = sol_cont.y[1] - sol_cont.y[4]
e_ddot_cont = sol_cont.y[2] - sol_cont.y[5]
z_cont = e_ddot_cont + lambda2 * e_dot_cont + lambda1 * e_cont
theta_hat_cont = sol_cont.y[6:12]

# --------------------------------------------------
# Tracking comparison: discrete vs continuous
# --------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
axes[0].plot(t, ym_vals, "--", color="gray", label=r"$y_m(t)$ (reference model)")
axes[0].plot(t, y_cont, label=r"$y(t)$ continuous")
axes[0].plot(t, y_vals, label=r"$y(t)$ discrete")
axes[0].set_ylabel("Output [rad]")
axes[0].set_title("Discrete vs. Continuous Adaptive Control: Output Tracking")
axes[0].grid(True)
axes[0].legend()
axes[1].plot(t, u_vals)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel(r"$u_k$")
axes[1].set_title("Discrete control input")
axes[1].grid(True)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "discrete_vs_continuous_tracking.png"), dpi=300)

# --------------------------------------------------
# Errors comparison: discrete vs continuous
# --------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
axes[0].plot(t, e_cont, label="continuous")
axes[0].plot(t, e_vals, label="discrete")
axes[0].set_ylabel(r"$e(t)$ [rad]")
axes[0].set_title("Discrete vs. Continuous Adaptive Control: Tracking Error")
axes[0].grid(True)
axes[0].legend()
axes[1].plot(t, z_cont, label="continuous")
axes[1].plot(t, z_vals, label="discrete")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel(r"$z(t)$")
axes[1].set_title("Filtered tracking error")
axes[1].grid(True)
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "discrete_vs_continuous_errors.png"), dpi=300)

# --------------------------------------------------
# Parameter estimates comparison: discrete vs continuous
# --------------------------------------------------
param_labels = [
    r"$\hat{a}_1$",
    r"$\hat{a}_2$",
    r"$\hat{a}_3$",
    r"$\hat{b}_1$",
    r"$\hat{b}_2$",
    r"$\hat{b}_3$",
]

fig, axes = plt.subplots(6, 1, figsize=(8, 11), sharex=True)
for i in range(6):
    axes[i].plot(t, theta_hat_cont[i], label="continuous")
    axes[i].plot(t, theta_hat_vals[i], label="discrete")
    axes[i].axhline(theta_true[i], color="gray", linestyle=":", label="true")
    axes[i].set_ylabel(param_labels[i])
    axes[i].grid(True)
    if i == 0:
        axes[i].legend(loc="best", fontsize=8)
axes[-1].set_xlabel("Time [s]")
axes[0].set_title("Discrete vs. Continuous Adaptive Control: Parameter Estimates")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "discrete_vs_continuous_parameters.png"), dpi=300)

# --------------------------------------------------
# ZOH detail: zoomed staircase view of u(t)
# --------------------------------------------------
zoom_end = 1.0  # seconds -- short enough to see individual steps
zoom_idx = np.searchsorted(t, zoom_end)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.step(t[:zoom_idx], u_vals[:zoom_idx], where="post", label=r"$u_k$ (ZOH)")
ax.plot(t[:zoom_idx], u_vals[:zoom_idx], "o", markersize=3, label="sample instants")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$u(t)$")
ax.set_title(
    f"Discrete Adaptive Control: ZOH detail "
    f"($T_s = {Ts}$ s, first {zoom_end:g} s)"
)
ax.grid(True)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "discrete_u_zoh_detail.png"), dpi=300)

plt.show()

# ==================================================
# Print simple metrics
# ==================================================
print(f"Sampling time Ts = {Ts} s")
print(f"Max |e|: {np.max(np.abs(e_vals)):.6f}")
print(f"Final |e|: {abs(e_vals[-1]):.6f}")
print("Final theta_hat vs. true:")
for label, est, tru in zip(param_labels, theta_hat_vals[:, -1], theta_true):
    print(f"  {label}: {est:+.4f}   (true {tru:+.4f}, err {est-tru:+.4f})")
