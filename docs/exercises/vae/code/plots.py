"""Helpers de figura compartilhados pelos Exercicios 1 e 2.

Centraliza o estilo dos eixos, a montagem de grades de digitos e a gravacao
dos arquivos em ``figures/``, para que as seis figuras do relatorio tenham a
mesma aparencia.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# --8<-- [start:style]
FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Uma cor por conjunto, sempre a mesma nas curvas.
COLORS = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
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


def tile(images: torch.Tensor, rows: int, cols: int, pad: int = 2) -> np.ndarray:
    """Junta rows * cols vetores de 784 em uma unica imagem, com borda entre eles."""

    images = images.detach().view(rows, cols, 28, 28).numpy()

    size = 28 + pad
    canvas = np.ones((rows * size + pad, cols * size + pad))

    for r in range(rows):
        for c in range(cols):
            top, left = pad + r * size, pad + c * size
            canvas[top : top + 28, left : left + 28] = images[r, c]

    return canvas


def show_digits(ax: plt.Axes, canvas: np.ndarray) -> None:
    """Desenha uma grade montada por tile(): fundo escuro, digito claro."""

    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


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
