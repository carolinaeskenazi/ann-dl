"""Helpers de figura compartilhados pelos Exercicios 2, 3 e 4.

Centraliza cores, o estilo dos eixos e a gravacao dos arquivos em
``figures/``, para que as seis figuras do relatorio tenham a mesma aparencia.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --8<-- [start:style]
FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Uma cor por classe, sempre na mesma ordem.
COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
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


# --8<-- [start:helpers]
def clean_axes(ax: plt.Axes) -> None:
    """Grade leve e sem as bordas de cima e da direita."""

    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_classes(
    ax: plt.Axes,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.55,
) -> None:
    """Desenha os pontos 2D, uma cor por classe."""

    for c in np.unique(y):
        ax.scatter(
            X[y == c, 0],
            X[y == c, 1],
            s=18,
            alpha=alpha,
            color=COLORS[int(c)],
            edgecolors="white",
            linewidths=0.3,
            label=f"Classe {c}",
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    clean_axes(ax)


def plot_loss(ax: plt.Axes, history: list[float], label: str, **style) -> None:
    """Curva de perda de treino por epoca."""

    ax.plot(range(1, len(history) + 1), history, label=label, **style)

    ax.set_xlabel("Época")
    ax.set_ylabel("Perda de treino")

    clean_axes(ax)


def save(fig: plt.Figure, filename: str) -> None:
    """Salva e fecha uma figura."""

    FIGURES.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURES / filename,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
# --8<-- [end:helpers]
