"""Exercise 1 — Point Clouds: Geometry and Spread in 2D.

Gera as quatro classes gaussianas do enunciado em quatro escalas de
dispersao, salva as Figuras 1, 2 e 3 em ``figures/`` e imprime as
metricas utilizadas no relatorio:

- separation ratio dos seis pares de classes em s = 1;
- menor separation ratio em s = 1 e seu valor em s = 2;
- mixing rate para s = 0.5, 1, 2 e 4.

Nenhum modelo e treinado neste exercicio.

Uso, a partir da raiz do repositorio:

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------

# --8<-- [start:config]
FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Semente fixa pedida pelo enunciado.
rng = np.random.default_rng(42)

CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}

N_PER_CLASS = 100
SCALES = (0.5, 1.0, 2.0, 4.0)

MEANS = np.array(
    [CLASSES[c]["mean"] for c in CLASSES],
    dtype=float,
)

STDS = np.array(
    [CLASSES[c]["std"] for c in CLASSES],
    dtype=float,
)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
# --8<-- [end:config]


# ---------------------------------------------------------------------
# Geracao dos dados
# ---------------------------------------------------------------------

# --8<-- [start:generate]
def generate(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Gera 100 pontos por classe com os desvios multiplicados por scale."""

    xs = []
    ys = []

    for label, params in CLASSES.items():
        mean = np.asarray(params["mean"], dtype=float)
        std = np.asarray(params["std"], dtype=float) * scale

        points = rng.normal(
            loc=mean,
            scale=std,
            size=(N_PER_CLASS, 2),
        )

        xs.append(points)
        ys.append(np.full(N_PER_CLASS, label, dtype=int))

    X = np.vstack(xs)
    y = np.concatenate(ys)

    return X, y
# --8<-- [end:generate]


# ---------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------

# --8<-- [start:metric-functions]
def separation_ratios(
    scale: float = 1.0,
) -> dict[tuple[int, int], float]:
    """Calcula r_ij para todos os seis pares de classes.

    r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j)

    em que sigma_bar_k e a media dos dois desvios padrao da classe k.
    """

    sigma_bar = STDS.mean(axis=1) * scale

    ratios = {}

    for i, j in combinations(CLASSES.keys(), 2):
        distance = np.linalg.norm(MEANS[i] - MEANS[j])

        ratios[(i, j)] = float(
            distance / (sigma_bar[i] + sigma_bar[j])
        )

    return ratios


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Calcula a fracao de pontos cujo centro mais proximo e de outra classe."""

    distances = np.linalg.norm(
        X[:, None, :] - MEANS[None, :, :],
        axis=2,
    )

    nearest_center = distances.argmin(axis=1)

    return float(np.mean(nearest_center != y))
# --8<-- [end:metric-functions]


# ---------------------------------------------------------------------
# Funcoes auxiliares para os graficos
# ---------------------------------------------------------------------

# --8<-- [start:scatter]
def scatter_classes(
    ax: plt.Axes,
    X: np.ndarray,
    y: np.ndarray,
    show_centers: bool = True,
) -> None:
    """Desenha os pontos de cada classe."""

    for c in CLASSES:
        ax.scatter(
            X[y == c, 0],
            X[y == c, 1],
            s=14,
            alpha=0.65,
            color=COLORS[c],
            label=f"Classe {c}",
        )

    if show_centers:
        ax.scatter(
            MEANS[:, 0],
            MEANS[:, 1],
            marker="X",
            s=140,
            color="black",
            edgecolors="white",
            linewidths=1.2,
            label="Centro (media)",
            zorder=5,
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
# --8<-- [end:scatter]


# --8<-- [start:boundaries]
def plot_boundaries(ax: plt.Axes) -> None:
    """Esboca regioes de decisao usando as distribuicoes conhecidas.

    Como as classes foram geradas por gaussianas com probabilidades
    a priori iguais, comparamos suas log-densidades. Isso serve apenas
    como esboco geometrico das fronteiras; nenhum modelo e treinado.
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 400),
        np.linspace(ylim[0], ylim[1], 400),
    )

    grid = np.column_stack(
        (xx.ravel(), yy.ravel())
    )

    # Distancia padronizada para cada classe.
    z = (
        grid[:, None, :] - MEANS[None, :, :]
    ) / STDS[None, :, :]

    # Log-densidade gaussiana.
    # O termo constante -log(2*pi) pode ser ignorado,
    # pois e igual para todas as classes.
    log_density = (
        -0.5 * (z ** 2).sum(axis=2)
        - np.log(STDS).sum(axis=1)
    )

    regions = log_density.argmax(axis=1).reshape(xx.shape)

    # Fundo suave indicando as regioes previstas.
    ax.contourf(
        xx,
        yy,
        regions,
        levels=np.arange(-0.5, len(CLASSES) + 0.5, 1),
        colors=COLORS,
        alpha=0.10,
    )

    # Desenha o contorno de cada regiao.
    for c in CLASSES:
        region_mask = (regions == c).astype(float)

        ax.contour(
            xx,
            yy,
            region_mask,
            levels=[0.5],
            colors="black",
            linewidths=1.0,
        )

    # Entrada manual para a legenda.
    ax.plot(
        [],
        [],
        color="black",
        linewidth=1.0,
        label="Fronteira esbocada",
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
# --8<-- [end:boundaries]


def save(fig: plt.Figure, filename: str) -> None:
    """Salva e fecha uma figura."""

    fig.savefig(
        FIGURES / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --8<-- [start:datasets]
    # -------------------------------------------------------------
    # Gera os quatro datasets pedidos no item B.
    # O mesmo objeto rng e utilizado em todas as geracoes.
    # -------------------------------------------------------------

    datasets = {
        scale: generate(scale)
        for scale in SCALES
    }

    # O dataset de s = 1 tambem representa o dataset original
    # utilizado no item A.
    X, y = datasets[1.0]
    # --8<-- [end:datasets]

    # --8<-- [start:metrics]
    # -------------------------------------------------------------
    # Metricas
    # -------------------------------------------------------------

    rates = {
        scale: mixing_rate(*datasets[scale])
        for scale in SCALES
    }

    ratios = separation_ratios(scale=1.0)

    min_pair, min_ratio = min(
        ratios.items(),
        key=lambda item: item[1],
    )

    # Como todas as dispersoes dobram de s=1 para s=2,
    # o separation ratio cai pela metade.
    min_ratio_s2 = min_ratio / 2.0
    # --8<-- [end:metrics]

    # --8<-- [start:fig1]
    # -------------------------------------------------------------
    # Figura 1 (item A)
    # Pontos + centros de cada classe.
    # -------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5.5))

    scatter_classes(ax, X, y)

    ax.set_title(
        "Figura 1 — Nuvens gaussianas (s = 1)"
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )

    fig.tight_layout()

    save(fig, "fig01-point-clouds.png")
    # --8<-- [end:fig1]

    # --8<-- [start:fig1-boundaries]
    # -------------------------------------------------------------
    # Figura 1 com o esboco das fronteiras (item C)
    # Mesmos pontos da Figura 1 + regioes de decisao.
    # -------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5.5))

    scatter_classes(ax, X, y)
    plot_boundaries(ax)

    ax.set_title(
        "Figura 1 — Fronteiras de decisao esbocadas (s = 1)"
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )

    fig.tight_layout()

    save(fig, "fig01-boundaries.png")
    # --8<-- [end:fig1-boundaries]

    # --8<-- [start:fig2]
    # -------------------------------------------------------------
    # Figura 2
    # Quatro datasets com exatamente os mesmos limites dos eixos.
    # -------------------------------------------------------------

    all_points = np.vstack(
        [datasets[scale][0] for scale in SCALES]
    )

    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    x_margin = 0.05 * (x_max - x_min)
    y_margin = 0.05 * (y_max - y_min)

    common_xlim = (
        x_min - x_margin,
        x_max + x_margin,
    )

    common_ylim = (
        y_min - y_margin,
        y_max + y_margin,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        sharex=True,
        sharey=True,
    )

    for ax, scale in zip(axes.flat, SCALES):
        Xs, ys = datasets[scale]

        scatter_classes(
            ax,
            Xs,
            ys,
            show_centers=True,
        )

        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)

        ax.set_title(
            f"s = {scale:g} | mixing rate = {rates[scale]:.2%}"
        )

        ax.tick_params(
            labelbottom=True,
            labelleft=True,
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
    )

    fig.suptitle(
        "Figura 2 — Efeito do fator de escala na dispersao das classes"
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    save(fig, "fig02-spread.png")
    # --8<-- [end:fig2]

    # --8<-- [start:fig3]
    # -------------------------------------------------------------
    # Figura 3
    # Mixing rate em funcao de s.
    # -------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4.5))

    mixing_values = [
        rates[scale]
        for scale in SCALES
    ]

    ax.plot(
        SCALES,
        mixing_values,
        marker="o",
        label="Taxa de mistura",
    )

    for scale in SCALES:
        ax.annotate(
            f"{rates[scale]:.2%}",
            (scale, rates[scale]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    ax.set_xscale("log", base=2)

    ax.set_xticks(
        SCALES,
        labels=[str(scale) for scale in SCALES],
    )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: f"{value:.0%}"
        )
    )

    ax.set_xlabel("Fator de escala $s$")
    ax.set_ylabel("Taxa de mistura")

    ax.set_title(
        "Figura 3 — Taxa de mistura em funcao do fator de escala"
    )

    ax.legend(loc="upper left")
    ax.margins(y=0.15)

    fig.tight_layout()

    save(fig, "fig03-mixing-rate.png")
    # --8<-- [end:fig3]

    # --8<-- [start:report]
    # -------------------------------------------------------------
    # Resultados numericos para colocar no relatorio.
    # -------------------------------------------------------------

    print("\nSEPARATION RATIO — s = 1\n")

    print("| Par | r_ij |")
    print("|-----|------|")

    for (i, j), ratio in ratios.items():
        print(f"| ({i}, {j}) | {ratio:.3f} |")

    print(
        f"\nMenor r_ij em s = 1: "
        f"par {min_pair} = {min_ratio:.3f}"
    )

    print(
        f"Mesmo par em s = 2, sem novo calculo geometrico: "
        f"{min_ratio:.3f} / 2 = {min_ratio_s2:.3f}"
    )

    print("\nMIXING RATE\n")

    for scale in SCALES:
        total = len(datasets[scale][1])

        n_mixed = int(
            np.sum(
                np.linalg.norm(
                    datasets[scale][0][:, None, :]
                    - MEANS[None, :, :],
                    axis=2,
                ).argmin(axis=1)
                != datasets[scale][1]
            )
        )

        print(
            f"s = {scale:g}: "
            f"{rates[scale]:.2%} "
            f"({n_mixed}/{total} pontos)"
        )
    # --8<-- [end:report]


if __name__ == "__main__":
    main()