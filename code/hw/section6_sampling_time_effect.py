"""
Section 6 -- Sampling-time effect and a Tustin (bilinear) redesign.

The Section 5 discrete controller estimates the error derivatives with raw
backward differences and integrates the adaptation law with forward Euler.
Both approximations lose accuracy as the sampling period Ts grows, and the
high-frequency gain of the second backward difference (~1/Ts^2), together with
the zero-order-hold delay, eventually destabilises the closed loop.

The improved design discretises the controller with the bilinear (Tustin)
transform:
  * the error derivatives are obtained from a "dirty derivative"
    H(s) = s/(tau*s + 1) realised with the bilinear transform, and
  * the adaptation law is integrated with the trapezoidal (Tustin) rule.

This better approximates the continuous-time controller, reduces the tracking
error at a given Ts, and extends the sampling period for which the closed loop
remains stable.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plot_utils import ensure_output_dir


# Plant parameters (true -- unknown to the controller)
a1, a2, a3 = 2.0, 3.0, 1.0
b1, b2, b3 = 0.05, 0.1, 1.5
c = 1.0
theta_true = np.array([a1, a2, a3, b1, b2, b3])

# Reference model parameters
zeta, wn = 0.7, 2.0
p3 = 10.0 * zeta * wn
alpha2 = 2.0 * zeta * wn + p3
alpha1 = wn**2 + 2.0 * zeta * wn * p3
alpha0 = wn**2 * p3

# Controller parameters
lambda1, lambda2 = 4.0, 4.0
kz = 5.0
Gamma = 10.0 * np.eye(6)
TAU = 0.06   # dirty-derivative filter time constant of the Tustin design


def r_cmd(t):
    """Constant step setpoint, pi/4 rad."""
    return np.pi / 4.0


def plant_derivative(t, xp, u_zoh):
    y, y_dot, y_ddot = xp
    y_3 = (
        -a1 * y_ddot - a2 * y_dot - a3 * y
        - b1 * y_ddot**3 - b2 * y_dot**3 - b3 * np.sin(y)
        + c * u_zoh
    )
    return np.array([y_dot, y_ddot, y_3])


def reference_model_derivative(t, xm):
    ym, ym_dot, ym_ddot = xm
    ym_3 = -alpha2 * ym_ddot - alpha1 * ym_dot - alpha0 * ym + alpha0 * r_cmd(t)
    return np.array([ym_dot, ym_ddot, ym_3])


# Sampled-data simulation
#
# method = "baseline": Section 5 controller
#          (backward-difference derivatives + forward-Euler adaptation)
# method = "tustin":   bilinear redesign
#          (dirty-derivative s/(tau s+1) via Tustin + trapezoidal adaptation)
def run_discrete_sim(Ts, method="baseline", tau=TAU, t_end=15.0,
                     diverge_threshold=1e3):
    num_steps = int(t_end / Ts)

    xp = np.zeros(3)
    xm = np.zeros(3)
    theta_hat = np.zeros(6)

    e_km1 = e_km2 = 0.0           # backward-difference history
    d1 = d1_prev = 0.0            # Tustin dirty-derivative state (e_dot)
    d2 = d2_prev = 0.0            # Tustin dirty-derivative state (e_ddot)
    phiz_prev = np.zeros(6)       # trapezoidal-adaptation history
    a = 2.0 / Ts                  # bilinear-transform constant

    diverged = False
    t_log, y_log, ym_log, e_log, z_log, u_log, theta_log = [], [], [], [], [], [], []

    for k in range(num_steps + 1):
        t_k = k * Ts
        y_k, y_dot_k, y_ddot_k = xp
        ym_k, ym_dot_k, ym_ddot_k = xm

        ym_3_k = (
            -alpha2 * ym_ddot_k - alpha1 * ym_dot_k - alpha0 * ym_k
            + alpha0 * r_cmd(t_k)
        )
        e_k = y_k - ym_k

        # error-derivative estimation
        if method == "baseline":
            # Raw backward differences (Section 5)
            if k == 0:
                edot, eddot = 0.0, 0.0
            elif k == 1:
                edot, eddot = (e_k - e_km1) / Ts, 0.0
            else:
                edot = (e_k - e_km1) / Ts
                eddot = (e_k - 2.0 * e_km1 + e_km2) / Ts**2
        else:
            # Bilinear (Tustin) dirty derivative: H(s) = s / (tau s + 1)
            # applied once for e_dot and twice (cascaded) for e_ddot.
            d1 = (a * (e_k - e_km1) - (1.0 - tau * a) * d1_prev) / (tau * a + 1.0)
            d2 = (a * (d1 - d1_prev) - (1.0 - tau * a) * d2_prev) / (tau * a + 1.0)
            edot, eddot = d1, d2
            d1_prev, d2_prev = d1, d2
        # -

        z_k = eddot + lambda2 * edot + lambda1 * e_k
        yr_3_k = ym_3_k - lambda2 * eddot - lambda1 * edot

        phi_k = np.array([
            y_ddot_k, y_dot_k, y_k, y_ddot_k**3, y_dot_k**3, np.sin(y_k),
        ])
        u_k = (theta_hat @ phi_k + yr_3_k - kz * z_k) / c

        # adaptation law
        phiz = (Gamma @ phi_k) * z_k
        if method == "baseline":
            theta_hat = theta_hat - Ts * phiz                       # forward Euler
        else:
            theta_hat = theta_hat - 0.5 * Ts * (phiz + phiz_prev)   # trapezoidal (Tustin)
        phiz_prev = phiz
        # -

        t_log.append(t_k); y_log.append(y_k); ym_log.append(ym_k)
        e_log.append(e_k); z_log.append(z_k); u_log.append(u_k)
        theta_log.append(theta_hat.copy())

        e_km2, e_km1 = e_km1, e_k

        if (not np.isfinite(theta_hat).all()
                or np.abs(theta_hat).max() > diverge_threshold
                or np.abs(xp).max() > diverge_threshold
                or abs(u_k) > diverge_threshold):
            diverged = True
            break

        if k < num_steps:
            sol_p = solve_ivp(lambda t, x: plant_derivative(t, x, u_k),
                              (t_k, t_k + Ts), xp, rtol=1e-7, atol=1e-9, max_step=Ts / 2)
            sol_m = solve_ivp(reference_model_derivative,
                              (t_k, t_k + Ts), xm, rtol=1e-7, atol=1e-9, max_step=Ts / 2)
            if not (sol_p.success and sol_m.success):
                diverged = True
                break
            xp = sol_p.y[:, -1]
            xm = sol_m.y[:, -1]

    return dict(
        t=np.array(t_log), y=np.array(y_log), ym=np.array(ym_log),
        e=np.array(e_log), z=np.array(z_log), u=np.array(u_log),
        theta=np.array(theta_log).T, diverged=diverged, Ts=Ts, method=method,
    )


# Run the simulations
TS_CMP = 0.05    # largest Ts where both controllers are stable -- fair comparison

deg = {Ts: run_discrete_sim(Ts, "baseline")
       for Ts in (0.01, 0.02, 0.03, 0.04, 0.05)}
base_cmp = run_discrete_sim(TS_CMP, "baseline")
tust_cmp = run_discrete_sim(TS_CMP, "tustin")

print("Performance vs sampling time  (max|e| for t>1s, '*'=diverged)")
print(f"{'Ts':>6} | {'baseline':>20} | {'Tustin':>20}")
for Ts in (0.01, 0.04, 0.05, 0.06, 0.07, 0.075, 0.08):
    rb = run_discrete_sim(Ts, "baseline")
    rt = run_discrete_sim(Ts, "tustin")

    def metric(r):
        m = np.max(np.abs(r["e"][r["t"] > 1.0])) if np.any(r["t"] > 1.0) else np.nan
        return f"{m:.4f}{'*' if r['diverged'] else ' '}"

    print(f"{Ts:6.3f} | {metric(rb):>20} | {metric(rt):>20}")


# Figures
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "report", "figs", "hw"
)
ensure_output_dir(output_dir)


# Fig 1: graceful degradation of the baseline controller
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for Ts in (0.01, 0.02, 0.03, 0.04, 0.05):
    r = deg[Ts]
    axes[0].plot(r["t"], r["e"], label=f"$T_s={Ts:.2f}$ s")
    axes[1].plot(r["t"], r["z"], label=f"$T_s={Ts:.2f}$ s")
axes[0].axhline(0.0, color="gray", linestyle=":", linewidth=1)   # perfect tracking
axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
axes[0].set_ylabel(r"$e(t)$ [rad]")
axes[0].set_title("Baseline controller at different sampling times")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel(r"$z(t)$")
for ax in axes:
    ax.grid(True); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "section6_degradation.png"), dpi=300)


# Fig 2: Tustin vs baseline at Ts=0.05 (full output, zoom, error)
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# full output
axes[0].plot(tust_cmp["t"], tust_cmp["ym"], ":", color="gray", label=r"$y_m(t)$")
axes[0].plot(base_cmp["t"], base_cmp["y"], color="C0", label="baseline")
axes[0].plot(tust_cmp["t"], tust_cmp["y"], color="C1", label="Tustin")
axes[0].set_ylabel("Output [rad]")
axes[0].set_title(fr"Baseline vs. Tustin at $T_s={TS_CMP:.2f}$ s")
axes[0].grid(True)
axes[0].legend(loc="lower right")

# zoom on the transient, where the two outputs differ most
dy = np.abs(base_cmp["y"] - tust_cmp["y"])
tp = base_cmp["t"][int(np.argmax(dy))]
x0, x1 = max(0.0, tp - 0.7), tp + 2.8
mask = (base_cmp["t"] >= x0) & (base_cmp["t"] <= x1)
ys = np.concatenate([base_cmp["y"][mask], tust_cmp["y"][mask], tust_cmp["ym"][mask]])
pad = 0.05 * (ys.max() - ys.min())
axes[1].plot(tust_cmp["t"], tust_cmp["ym"], ":", color="gray", label=r"$y_m(t)$")
axes[1].plot(base_cmp["t"], base_cmp["y"], color="C0", label="baseline")
axes[1].plot(tust_cmp["t"], tust_cmp["y"], color="C1", label="Tustin")
axes[1].set_xlim(x0, x1)
axes[1].set_ylim(ys.min() - pad, ys.max() + pad)
axes[1].set_ylabel("Output [rad]")
axes[1].set_title("Zoomed-in")
axes[1].grid(True)
axes[1].legend(loc="lower right")

# tracking error
axes[2].plot(base_cmp["t"], base_cmp["e"], color="C0", label="baseline")
axes[2].plot(tust_cmp["t"], tust_cmp["e"], color="C1", label="Tustin redesign")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel(r"$e(t)$ [rad]")
axes[2].grid(True)
axes[2].legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "section6_tustin_fix.png"), dpi=300)


plt.show()
