# --------------------
# Section 3
# --------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plot_utils import (
    ensure_output_dir,
    plot_tracking_and_control,
    plot_errors,
)

# Plant parameters
a1 = 2.0
a2 = 3.0
a3 = 1.0
b1 = 0.05
b2 = 0.1
b3 = 1.5
c = 1.0

theta = np.array([a1, a2, a3, b1, b2, b3])

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


# Reference input
def r(t):
    return np.pi / 4

# Combined plant + reference model dynamics
# State vector:
# X = [y, y_dot, y_ddot, ym, ym_dot, ym_ddot]
def closed_loop_dynamics(t, X):
    y, y_dot, y_ddot, ym, ym_dot, ym_ddot = X

    # Reference model third derivative
    ym_3 = (
        -alpha2 * ym_ddot
        -alpha1 * ym_dot
        -alpha0 * ym
        + alpha0 * r(t)
    )

    # Tracking errors
    e = y - ym
    e_dot = y_dot - ym_dot
    e_ddot = y_ddot - ym_ddot

    # Filtered tracking error
    z = e_ddot + lambda2 * e_dot + lambda1 * e

    # Auxiliary reference signal
    yr_3 = ym_3 - lambda2 * e_ddot - lambda1 * e_dot

    # Regressor vector
    phi = np.array([
        y_ddot,
        y_dot,
        y,
        y_ddot**3,
        y_dot**3,
        np.sin(y)
    ])

    # Nominal control law
    u = (theta @ phi + yr_3 - kz * z) / c

    # Plant third derivative
    y_3 = (
        -a1 * y_ddot
        -a2 * y_dot
        -a3 * y
        -b1 * y_ddot**3
        -b2 * y_dot**3
        -b3 * np.sin(y)
        + c * u
    )

    return [
        y_dot,
        y_ddot,
        y_3,
        ym_dot,
        ym_ddot,
        ym_3
    ]


# Simulation
t_start = 0.0
t_end = 20.0
t_eval = np.linspace(t_start, t_end, 3000)

# Initial conditions
X0 = np.array([
    0.0,  # y
    0.0,  # y_dot
    0.0,  # y_ddot
    0.0,  # ym
    0.0,  # ym_dot
    0.0   # ym_ddot
])

sol = solve_ivp(
    closed_loop_dynamics,
    (t_start, t_end),
    X0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10
)

if not sol.success:
    raise RuntimeError("ODE solver failed: " + sol.message)

# Post-processing
t = sol.t
y = sol.y[0]
y_dot = sol.y[1]
y_ddot = sol.y[2]
ym = sol.y[3]
ym_dot = sol.y[4]
ym_ddot = sol.y[5]

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
        + alpha0 * r(ti)
    )

    yr_3 = ym_3 - lambda2 * e_ddot[i] - lambda1 * e_dot[i]

    phi = np.array([
        y_ddot[i],
        y_dot[i],
        y[i],
        y_ddot[i]**3,
        y_dot[i]**3,
        np.sin(y[i])
    ])

    u_values[i] = (theta @ phi + yr_3 - kz * z[i]) / c


# Save figures
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "report", "figs", "hw"
)
ensure_output_dir(output_dir)

r_values = np.array([r(ti) for ti in t])

plot_tracking_and_control(
    t, r_values, ym, y, u_values,
    output_dir, "section3_nominal_tracking_and_control.png",
    title_prefix="Nominal Control",
)

plot_errors(
    t, e, z,
    output_dir, "section3_nominal_errors.png",
    title_prefix="Nominal Control",
)

plt.show()