import os
import matplotlib.pyplot as plt


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_tracking_and_control(
    t, r_values, ym, y, u, output_dir, filename, title_prefix
):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, r_values, ":", color="gray", label=r"$r(t)$")
    axes[0].plot(t, ym, "--", label=r"$y_m(t)$")
    axes[0].plot(t, y, label=r"$y(t)$")
    axes[0].set_ylabel("Output [rad]")
    axes[0].set_title(f"{title_prefix}: Output Tracking")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(t, u)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$u(t)$")
    axes[1].set_title(f"{title_prefix}: Control Input")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
    return fig


def plot_errors(t, e, z, output_dir, filename, title_prefix):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, e)
    axes[0].set_ylabel(r"$e(t)$ [rad]")
    axes[0].set_title(f"{title_prefix}: Tracking Error")
    axes[0].grid(True)
    axes[1].plot(t, z)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$z(t)$")
    axes[1].set_title(f"{title_prefix}: Filtered Tracking Error")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
    return fig


def plot_parameter_estimates(
    t, theta_hat, theta_true, param_labels, output_dir, filename, title_prefix
):
    n = len(param_labels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n), sharex=True)
    for i in range(n):
        axes[i].plot(t, theta_hat[i], label=param_labels[i])
        axes[i].axhline(
            theta_true[i], linestyle="--", color="gray", label="true"
        )
        axes[i].set_ylabel(param_labels[i])
        axes[i].grid(True)
        axes[i].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"{title_prefix}: Parameter Estimates")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
    return fig
