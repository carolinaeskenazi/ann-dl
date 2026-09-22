"""Helpers de figura compartilhados pelos dois exercícios.

Centraliza cores, o desenho das nuvens de pontos, o desenho de uma
fronteira de decisão w . x + b = 0 e a gravação dos arquivos em
``figures/``, para que as seis figuras do relatório tenham a mesma aparência.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuração visual
# ---------------------------------------------------------------------

# --8<-- [start:style]

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Uma cor por classe, sempre na mesma ordem.
COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
}

# Fronteiras e curvas usam também estilos de linha diferentes.
FINAL_STYLE = {
    "color": "#111111",
    "linestyle": "--",
    "linewidth": 2.5,
}

POCKET_STYLE = {
    "color": "#d62728",
    "linestyle": "-",
    "linewidth": 2.5,
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)

# --8<-- [end:style]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# --8<-- [start:helpers]

def scatter_classes(
    ax: plt.Axes,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Desenha os pontos, uma cor por classe."""

    for c in (0, 1):
        ax.scatter(
            X[y == c, 0],
            X[y == c, 1],
            s=14,
            alpha=0.35,
            color=COLORS[c],
            edgecolors="none",
            label=f"Classe {c}",
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_boundary(
    ax: plt.Axes,
    w: np.ndarray,
    b: float,
    label: str,
    **style,
) -> None:
    """Desenha a reta w1*x1 + w2*x2 + b = 0 dentro dos limites do eixo.

    Quando w2 é praticamente zero, a fronteira é vertical.
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if abs(w[1]) < 1e-12:
        ax.axvline(
            -b / w[0],
            label=label,
            **style,
        )
    else:
        x1 = np.linspace(xlim[0], xlim[1], 300)
        x2 = -(w[0] * x1 + b) / w[1]

        ax.plot(
            x1,
            x2,
            label=label,
            **style,
        )

    # A reta não deve mudar o enquadramento dos dados.
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def mark_errors(
    ax: plt.Axes,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> int:
    """Marca com X os pontos classificados errado."""

    wrong = y_pred != y

    ax.scatter(
        X[wrong, 0],
        X[wrong, 1],
        s=42,
        marker="x",
        color="#111111",
        linewidths=1.1,
        label=f"{label} ({int(wrong.sum())})",
        zorder=4,
    )

    return int(wrong.sum())


def save(
    fig: plt.Figure,
    filename: str,
) -> None:
    """Salva e fecha uma figura."""

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURES / filename,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

# --8<-- [end:helpers]