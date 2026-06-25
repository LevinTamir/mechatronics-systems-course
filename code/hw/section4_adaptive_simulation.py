# --------------------
# Section 4
# --------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plot_utils import (
    ensure_output_dir,
    plot_tracking_and_control,
    plot_errors,
    plot_parameter_estimates,
)


# Plant parameters (true — unknown to the controller)
a1 = 2.0
a2 = 3.0
a3 = 1.0
b1 = 0.05
b2 = 0.1
b3 = 1.5
c = 1.0

theta_true = np.array([a1, a2, a3, b1, b2, b3])


# Reference model parameters
zeta = 0.7
wn = 2.0
p3 = 10.0 * zeta * wn

alpha2 = 2.0 * zeta * wn + p3
alpha1 = wn**2 + 2.0 * zeta * wn * p3
alpha0 = wn**2 * p3

print("Reference model parameters:")
print(f"alpha2 = {alpha2:.4f}")
print(f"alpha1 = {alpha1:.4f}")
print(f"alpha0 = {alpha0:.4f}")


# Controller parameters
lambda1 = 4.0
lambda2 = 4.0
kz = 5.0

# Adaptation gain
Gamma = 10.0 * np.eye(6)


# Reference input variants
def r_step(t):
    return np.pi / 4


def r_pe(t):
    return (
        0.4
        + 0.2 * np.sin(0.5 * t)
        + 0.15 * np.sin(1.3 * t)
        + 0.1 * np.sin(2.1 * t)
    )


# Combined plant + reference model + adaptation dynamics
# State vector (12):
# X = [y, y_dot, y_ddot, ym, ym_dot, ym_ddot,
#      theta_hat_1, ..., theta_hat_6]
def make_dynamics(r_func):
    def closed_loop_dynamics(t, X):
        y, y_dot, y_ddot, ym, ym_dot, ym_ddot = X[:6]
        theta_hat = X[6:12]

        ym_3 = (
            -alpha2 * ym_ddot
            -alpha1 * ym_dot
            -alpha0 * ym
            + alpha0 * r_func(t)
        )

        e = y - ym
        e_dot = y_dot - ym_dot
        e_ddot = y_ddot - ym_ddot

        z = e_ddot + lambda2 * e_dot + lambda1 * e

        yr_3 = ym_3 - lambda2 * e_ddot - lambda1 * e_dot

        phi = np.array([
            y_ddot,
            y_dot,
            y,
            y_ddot**3,
            y_dot**3,
            np.sin(y)
        ])

        u = (theta_hat @ phi + yr_3 - kz * z) / c

        y_3 = (
            -a1 * y_ddot
            -a2 * y_dot
            -a3 * y
            -b1 * y_ddot**3
            -b2 * y_dot**3
            -b3 * np.sin(y)
            + c * u
        )

        theta_hat_dot = -Gamma @ phi * z

        return [
            y_dot,
            y_ddot,
            y_3,
            ym_dot,
            ym_ddot,
            ym_3,
            theta_hat_dot[0],
            theta_hat_dot[1],
            theta_hat_dot[2],
            theta_hat_dot[3],
            theta_hat_dot[4],
            theta_hat_dot[5],
        ]
    return closed_loop_dynamics


# Run one simulation for a given reference and save figures
def run_simulation(r_func, suffix, title_prefix, t_end):
    t_eval = np.linspace(0.0, t_end, 3000)
    X0 = np.zeros(12)

    sol = solve_ivp(
        make_dynamics(r_func),
        (0.0, t_end),
        X0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)

    t = sol.t
    y = sol.y[0]
    y_dot = sol.y[1]
    y_ddot = sol.y[2]
    ym = sol.y[3]
    ym_dot = sol.y[4]
    ym_ddot = sol.y[5]
    theta_hat = sol.y[6:12]

    e = y - ym
    e_dot = y_dot - ym_dot
    e_ddot = y_ddot - ym_ddot
    z = e_ddot + lambda2 * e_dot + lambda1 * e

    u_values = np.zeros_like(t)
    for i in range(len(t)):
        ti = t[i]

        ym_3 = (
            -alpha2 * ym_ddot[i]
            -alpha1 * ym_dot[i]
            -alpha0 * ym[i]
            + alpha0 * r_func(ti)
        )
        yr_3 = ym_3 - lambda2 * e_ddot[i] - lambda1 * e_dot[i]

        phi = np.array([
            y_ddot[i],
            y_dot[i],
            y[i],
            y_ddot[i]**3,
            y_dot[i]**3,
            np.sin(y[i]),
        ])

        u_values[i] = (theta_hat[:, i] @ phi + yr_3 - kz * z[i]) / c

    r_values = np.array([r_func(ti) for ti in t])

    plot_tracking_and_control(
        t, r_values, ym, y, u_values,
        output_dir, f"section4_adaptive_tracking_and_control_{suffix}.png",
        title_prefix=title_prefix,
    )

    plot_errors(
        t, e, z,
        output_dir, f"section4_adaptive_errors_{suffix}.png",
        title_prefix=title_prefix,
    )

    plot_parameter_estimates(
        t, theta_hat, theta_true, param_labels,
        output_dir, f"section4_adaptive_parameter_estimates_{suffix}.png",
        title_prefix=title_prefix,
    )

    print(f"[{suffix}] Final parameter estimates vs. truth:")
    for label, est, tru in zip(param_labels, theta_hat[:, -1], theta_true):
        print(f"  {label}: {est:+.4f}   (true {tru:+.4f}, err {est-tru:+.4f})")


# Run both cases
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "report", "figs", "hw"
)
ensure_output_dir(output_dir)

param_labels = [
    r"$\hat{a}_1$",
    r"$\hat{a}_2$",
    r"$\hat{a}_3$",
    r"$\hat{b}_1$",
    r"$\hat{b}_2$",
    r"$\hat{b}_3$",
]

# Case 1: constant step reference (no PE)
run_simulation(
    r_step,
    suffix="step",
    title_prefix="Adaptive Control (Step Reference)",
    t_end=20.0,
)

# Case 2: PE-rich multi-sinusoidal reference
run_simulation(
    r_pe,
    suffix="pe",
    title_prefix="Adaptive Control (PE Reference)",
    t_end=40.0,
)

plt.show()
