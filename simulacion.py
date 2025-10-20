import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def kuramoto(N, omega, K, dt, steps, theta0=None):
    """
    Modelo de Kuramoto:
        dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)
    """
    if theta0 is None:
        theta = np.random.uniform(0, 2 * np.pi, N)
    else:
        theta = theta0.copy()

    history = np.zeros((steps, N))
    R = np.zeros(steps)

    for t in range(steps):
        history[t] = theta % (2 * np.pi)
        order = np.mean(np.exp(1j * theta))
        R[t] = np.abs(order)
        theta_dot = omega + (K / N) * np.sum(
            np.sin(theta[None, :] - theta[:, None]), axis=1
        )
        theta = theta + dt * theta_dot

    return history, R


def winfree(N, omega, eps, P_func, Q_func, dt, steps, theta0=None):
    """
    Modelo de Winfree:
        dθ_i/dt = ω_i + ε * Q(θ_i) * ⟨P(θ_j)⟩
    """
    if theta0 is None:
        theta = np.random.uniform(0, 2 * np.pi, N)
    else:
        theta = theta0.copy()

    history = np.zeros((steps, N))
    R = np.zeros(steps)

    for t in range(steps):
        history[t] = theta % (2 * np.pi)
        order = np.mean(np.exp(1j * theta))
        R[t] = np.abs(order)
        P_vals = P_func(theta)
        meanP = np.mean(P_vals)
        theta_dot = omega + eps * Q_func(theta) * meanP
        theta = theta + dt * theta_dot

    return history, R


def P(theta):
    """Pulso emitido por cada oscilador"""
    return 1 + np.cos(theta)


def Q(theta):
    """Sensibilidad del oscilador"""
    return -np.sin(theta)


if __name__ == "__main__":
    N = 50  # Número de osciladores
    dt = 0.05  # Paso de integración
    T = 50.0  # Tiempo total
    steps = int(T / dt)
    np.random.seed(42)  # Reproducibilidad

    omega = np.random.normal(0.0, 1.0, N)  # Frecuencias naturales
    theta0 = np.random.uniform(0, 2 * np.pi, N)  # Condiciones iniciales

    K = 2.0  # Acoplamiento Kuramoto
    eps = 1.0  # Acoplamiento Winfree

    hist_k, R_k = kuramoto(N, omega, K, dt, steps, theta0)
    hist_w, R_w = winfree(N, omega, eps, P, Q, dt, steps, theta0)

    times = np.linspace(0, T, steps)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Comparación de Modelos de Sincronización: Kuramoto vs Winfree",
        fontsize=14,
        weight="bold",
    )

    for i in range(N):
        axs[0, 0].plot(times, hist_k[:, i], lw=0.8, alpha=0.7)
    axs[0, 0].set_title("Kuramoto – Evolución de las fases θᵢ(t)")
    axs[0, 0].set_xlabel("Tiempo")
    axs[0, 0].set_ylabel("Fase (rad)")
    axs[0, 0].set_ylim(0, 2 * np.pi)
    axs[0, 0].grid(True, linestyle="--", alpha=0.4)

    axs[0, 1].plot(times, R_k, lw=2, color="tab:blue")
    axs[0, 1].set_title("Kuramoto – Orden colectivo R(t)")
    axs[0, 1].set_xlabel("Tiempo")
    axs[0, 1].set_ylabel("R(t)")
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].grid(True, linestyle="--", alpha=0.4)
    axs[0, 1].annotate(
        "Mayor sincronización",
        xy=(T * 0.8, R_k[-1]),
        xytext=(T * 0.5, 0.3),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    for i in range(N):
        axs[1, 0].plot(times, hist_w[:, i], lw=0.8, alpha=0.7)
    axs[1, 0].set_title("Winfree – Evolución de las fases θᵢ(t)")
    axs[1, 0].set_xlabel("Tiempo")
    axs[1, 0].set_ylabel("Fase (rad)")
    axs[1, 0].set_ylim(0, 2 * np.pi)
    axs[1, 0].grid(True, linestyle="--", alpha=0.4)

    axs[1, 1].plot(times, R_w, lw=2, color="tab:orange")
    axs[1, 1].set_title("Winfree – Orden colectivo R(t)")
    axs[1, 1].set_xlabel("Tiempo")
    axs[1, 1].set_ylabel("R(t)")
    axs[1, 1].set_ylim(0, 1.05)
    axs[1, 1].grid(True, linestyle="--", alpha=0.4)
    axs[1, 1].annotate(
        "Transición a sincronía",
        xy=(T * 0.8, R_w[-1]),
        xytext=(T * 0.4, 0.4),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    df = pd.DataFrame(
        {
            "Modelo": ["Kuramoto", "Winfree"],
            "R_final": [float(R_k[-1]), float(R_w[-1])],
            "R_promedio_último_10%": [
                float(np.mean(R_k[int(0.9 * steps) :])),
                float(np.mean(R_w[int(0.9 * steps) :])),
            ],
        }
    )

    print("\nComparación entre modelos:")
    print(df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.plot(times, R_k, label="Kuramoto", lw=2, color="tab:blue")
    plt.plot(times, R_w, label="Winfree", lw=2, color="tab:orange", linestyle="--")
    plt.title(
        "Comparación del Orden Colectivo R(t) entre Modelos", fontsize=13, weight="bold"
    )
    plt.xlabel("Tiempo")
    plt.ylabel("R(t)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    plt.annotate(
        "Kuramoto alcanza sincronía estable",
        xy=(T * 0.8, R_k[-1]),
        xytext=(T * 0.5, 0.85),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="tab:blue"),
        color="tab:blue",
    )

    plt.annotate(
        "Winfree logra mayor sincronía final",
        xy=(T * 0.8, R_w[-1]),
        xytext=(T * 0.3, 0.6),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="tab:orange"),
        color="tab:orange",
    )

    plt.tight_layout()
    plt.show()
