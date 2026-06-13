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
# Reference: step setpoint
# ==================================================
def r_cmd(t):
    return np.pi / 4


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
# Sampled-data simulation
#
# use_filter=False: raw backward differences for e_dot, e_ddot
# use_filter=True : 3rd-order state-variable filter (cutoff omega_f)
#                   that produces smooth (e_hat, e_dot_hat, e_ddot_hat)
#                   from samples of e.  This is the standard textbook
#                   replacement for direct numerical differentiation.
# ==================================================
def run_discrete_sim(Ts, use_filter=False, omega_f=15.0,
                     t_end=15.0, diverge_threshold=1e3):
    num_steps = int(t_end / Ts)

    xp = np.zeros(3)
    xm = np.zeros(3)
    theta_hat_k = np.zeros(6)

    # State-variable filter states: [e_hat, e_dot_hat, e_ddot_hat]
    sv = np.zeros(3)

    # Backward-difference history
    e_km1 = 0.0
    e_km2 = 0.0

    u_k = 0.0
    diverged = False

    t_log, e_log, z_log, u_log, theta_log = [], [], [], [], []

    for k in range(num_steps + 1):
        t_k = k * Ts

        y_k, y_dot_k, y_ddot_k = xp
        ym_k, ym_dot_k, ym_ddot_k = xm

        ym_3_k = (
            -alpha2 * ym_ddot_k
            -alpha1 * ym_dot_k
            -alpha0 * ym_k
            + alpha0 * r_cmd(t_k)
        )

        e_k = y_k - ym_k

        # ---------- derivative estimation ----------
        if use_filter:
            # 3rd-order LP filter: 1 / (s + omega_f)^3 applied to e
            # State: [e_hat, e_dot_hat, e_ddot_hat]
            e3_dot = (
                - 3.0 * omega_f * sv[2]
                - 3.0 * omega_f**2 * sv[1]
                - omega_f**3 * sv[0]
                + omega_f**3 * e_k
            )
            sv_new = np.array([
                sv[0] + Ts * sv[1],
                sv[1] + Ts * sv[2],
                sv[2] + Ts * e3_dot,
            ])
            sv = sv_new
            e_used    = sv[0]
            edot_used = sv[1]
            eddot_used= sv[2]
        else:
            # Raw backward differences
            if k == 0:
                edot_used = 0.0
                eddot_used = 0.0
            elif k == 1:
                edot_used = (e_k - e_km1) / Ts
                eddot_used = 0.0
            else:
                edot_used = (e_k - e_km1) / Ts
                eddot_used = (e_k - 2.0 * e_km1 + e_km2) / Ts**2
            e_used = e_k
        # -------------------------------------------

        z_k = eddot_used + lambda2 * edot_used + lambda1 * e_used
        yr_3_k = ym_3_k - lambda2 * eddot_used - lambda1 * edot_used

        phi_k = np.array([
            y_ddot_k, y_dot_k, y_k,
            y_ddot_k**3, y_dot_k**3, np.sin(y_k),
        ])

        u_k = (theta_hat_k @ phi_k + yr_3_k - kz * z_k) / c

        # Forward-Euler adaptation
        theta_hat_k = theta_hat_k - Ts * (Gamma @ phi_k) * z_k

        t_log.append(t_k)
        e_log.append(e_k)
        z_log.append(z_k)
        u_log.append(u_k)
        theta_log.append(theta_hat_k.copy())

        e_km2 = e_km1
        e_km1 = e_k

        # Divergence guard
        if (
            not np.isfinite(theta_hat_k).all()
            or not np.isfinite(xp).all()
            or np.abs(theta_hat_k).max() > diverge_threshold
            or np.abs(xp).max() > diverge_threshold
            or abs(u_k) > diverge_threshold
        ):
            diverged = True
            break

        if k < num_steps:
            t_next = t_k + Ts
            sol_p = solve_ivp(
                lambda t, x: plant_derivative(t, x, u_k),
                (t_k, t_next), xp,
                rtol=1e-6, atol=1e-8, max_step=Ts / 2,
            )
            sol_m = solve_ivp(
                reference_model_derivative,
                (t_k, t_next), xm,
                rtol=1e-6, atol=1e-8, max_step=Ts / 2,
            )
            if not (sol_p.success and sol_m.success):
                diverged = True
                break
            xp = sol_p.y[:, -1]
            xm = sol_m.y[:, -1]

    return {
        "t": np.array(t_log),
        "e": np.array(e_log),
        "z": np.array(z_log),
        "u": np.array(u_log),
        "theta": np.array(theta_log).T,
        "diverged": diverged,
    }


# ==================================================
# Configurations to compare
# ==================================================
DEGRADED_TS = 0.04

sims = {
    "Ts=0.01 (baseline)":             run_discrete_sim(0.01, use_filter=False),
    "Ts=0.02 (raw)":                  run_discrete_sim(0.02, use_filter=False),
    f"Ts={DEGRADED_TS:.2f} (raw)":    run_discrete_sim(DEGRADED_TS, use_filter=False),
    f"Ts={DEGRADED_TS:.2f} (filter)": run_discrete_sim(DEGRADED_TS, use_filter=True),
}

print("\nSummary of final |e| and max |e|:")
for name, r in sims.items():
    tag = "(DIVERGED)" if r["diverged"] else ""
    print(
        f"  {name:30s}  max|e|={np.max(np.abs(r['e'])):.4f}  "
        f"|e_final|={abs(r['e'][-1]):.4f}  {tag}"
    )


# ==================================================
# Plot 1: degradation with increasing Ts (Euler only)
# ==================================================
output_dir = "figs/hw/"
ensure_output_dir(output_dir)

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for name in [
    "Ts=0.01 (baseline)",
    "Ts=0.02 (raw)",
    f"Ts={DEGRADED_TS:.2f} (raw)",
]:
    r = sims[name]
    axes[0].plot(r["t"], r["e"], label=name)
    axes[1].plot(r["t"], r["z"], label=name)
axes[0].set_ylabel(r"$e(t)$ [rad]")
axes[0].set_title("Performance degradation with larger $T_s$ (raw backward differences)")
axes[0].grid(True)
axes[0].legend()
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel(r"$z(t)$")
axes[1].grid(True)
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "section6_degradation.png"), dpi=300)


# ==================================================
# Plot 2: state-variable filter recovers performance
# ==================================================
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for name in [
    "Ts=0.01 (baseline)",
    f"Ts={DEGRADED_TS:.2f} (raw)",
    f"Ts={DEGRADED_TS:.2f} (filter)",
]:
    r = sims[name]
    axes[0].plot(r["t"], r["e"], label=name)
    axes[1].plot(r["t"], r["z"], label=name)
axes[0].set_ylabel(r"$e(t)$ [rad]")
axes[0].set_title(
    f"Effect of state-variable filter at $T_s = {DEGRADED_TS:.2f}$ s"
)
axes[0].grid(True)
axes[0].legend()
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel(r"$z(t)$")
axes[1].grid(True)
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "section6_filter_fix.png"), dpi=300)


# ==================================================
# Plot 3: parameter estimates at degraded vs improved
# ==================================================
param_labels = [
    r"$\hat{a}_1$", r"$\hat{a}_2$", r"$\hat{a}_3$",
    r"$\hat{b}_1$", r"$\hat{b}_2$", r"$\hat{b}_3$",
]

fig, axes = plt.subplots(6, 1, figsize=(8, 11), sharex=True)
for i in range(6):
    for name, ls in [
        ("Ts=0.01 (baseline)",             "-"),
        (f"Ts={DEGRADED_TS:.2f} (raw)",    "--"),
        (f"Ts={DEGRADED_TS:.2f} (filter)", "-."),
    ]:
        r = sims[name]
        axes[i].plot(r["t"], r["theta"][i], ls, label=name if i == 0 else None)
    axes[i].axhline(theta_true[i], color="gray", linestyle=":", linewidth=1)
    axes[i].set_ylabel(param_labels[i])
    axes[i].grid(True)
axes[0].legend(loc="best", fontsize=8)
axes[0].set_title("Parameter estimates: baseline vs. degraded vs. filtered")
axes[-1].set_xlabel("Time [s]")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "section6_parameter_comparison.png"), dpi=300)

plt.show()
