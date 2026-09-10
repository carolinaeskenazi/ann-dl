"""Exercise 2 — Non-Linearity in Higher Dimensions.

Gera dois datasets 5D:
1. Duas gaussianas deslocadas;
2. Duas cascas concentricas.

Aplica PCA para visualizacao, calcula as distancias entre centros
em 5D e gera as Figuras 4 e 5.

Nenhum modelo e treinado.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


FIGURES = Path(__file__).resolve().parents[1] / "figures"

rng = np.random.default_rng(42)

N_PER_CLASS = 500
COLORS = ["tab:blue", "tab:orange"]


# --8<-- [start:dataset1]
# ---------------------------------------------------------------------
# Dataset I — shifted Gaussians
# ---------------------------------------------------------------------

MU_A = np.array([0, 0, 0, 0, 0], dtype=float)

MU_B = np.array(
    [1.5, 1.5, 1.5, 1.5, 1.5],
    dtype=float,
)

SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])


def generate_dataset_i():
    """Gera as classes A e B usando gaussianas multivariadas."""

    class_a = rng.multivariate_normal(
        MU_A,
        SIGMA_A,
        size=N_PER_CLASS,
    )

    class_b = rng.multivariate_normal(
        MU_B,
        SIGMA_B,
        size=N_PER_CLASS,
    )

    X = np.vstack([class_a, class_b])

    y = np.array(
        [0] * N_PER_CLASS
        + [1] * N_PER_CLASS
    )

    return X, y, class_a, class_b
# --8<-- [end:dataset1]


# --8<-- [start:dataset2]
# ---------------------------------------------------------------------
# Dataset II — concentric shells
# ---------------------------------------------------------------------

def generate_shell(radius_mean):
    """Gera uma casca em R^5 com direcoes uniformes na esfera."""

    # Vetores gaussianos em R^5.
    v = rng.normal(
        size=(N_PER_CLASS, 5)
    )

    # Normalizacao para obter direcoes na esfera unitaria.
    u = v / np.linalg.norm(
        v,
        axis=1,
        keepdims=True,
    )

    # Raio da casca.
    rho = rng.normal(
        radius_mean,
        0.4,
        size=N_PER_CLASS,
    )

    return rho[:, None] * u


def generate_dataset_ii():
    """Gera a classe C (core) e a classe D (shell)."""

    class_c = generate_shell(2.0)
    class_d = generate_shell(5.0)

    X = np.vstack([
        class_c,
        class_d,
    ])

    y = np.array(
        [0] * N_PER_CLASS
        + [1] * N_PER_CLASS
    )

    return X, y, class_c, class_d
# --8<-- [end:dataset2]


# --8<-- [start:center-distance]
def center_distance(class_1, class_2):
    """Distancia euclidiana entre os centroides em 5D."""

    center_1 = class_1.mean(axis=0)
    center_2 = class_2.mean(axis=0)

    return float(
        np.linalg.norm(
            center_1 - center_2
        )
    )
# --8<-- [end:center-distance]


def save(fig, filename):
    fig.savefig(
        FIGURES / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Gera os dois datasets.
    # -------------------------------------------------------------

    X1, y1, class_a, class_b = generate_dataset_i()
    X2, y2, class_c, class_d = generate_dataset_ii()

    # --8<-- [start:pca]
    # -------------------------------------------------------------
    # PCA
    # -------------------------------------------------------------

    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)

    Z1 = pca1.fit_transform(X1)
    Z2 = pca2.fit_transform(X2)

    explained_1 = float(
        pca1.explained_variance_ratio_.sum()
    )

    explained_2 = float(
        pca2.explained_variance_ratio_.sum()
    )
    # --8<-- [end:pca]

    # --8<-- [start:distances]
    # -------------------------------------------------------------
    # Distancia entre centros no espaco original 5D.
    # -------------------------------------------------------------

    distance_1 = center_distance(
        class_a,
        class_b,
    )

    distance_2 = center_distance(
        class_c,
        class_d,
    )
    # --8<-- [end:distances]

    # --8<-- [start:fig4]
    # -------------------------------------------------------------
    # Figura 4 — PCA
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    for label, name in [
        (0, "Classe A"),
        (1, "Classe B"),
    ]:
        axes[0].scatter(
            Z1[y1 == label, 0],
            Z1[y1 == label, 1],
            s=14,
            alpha=0.6,
            color=COLORS[label],
            label=name,
        )

    axes[0].set_title(
        f"Dataset I — PCA "
        f"({explained_1:.2%} de variância explicada)"
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    for label, name in [
        (0, "Classe C"),
        (1, "Classe D"),
    ]:
        axes[1].scatter(
            Z2[y2 == label, 0],
            Z2[y2 == label, 1],
            s=14,
            alpha=0.6,
            color=COLORS[label],
            label=name,
        )

    axes[1].set_title(
        f"Dataset II — PCA "
        f"({explained_2:.2%} de variância explicada)"
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    fig.suptitle(
        "Figura 4 — Projeção PCA dos dois datasets"
    )

    fig.tight_layout()

    save(
        fig,
        "fig04-pca.png",
    )
    # --8<-- [end:fig4]

    # --8<-- [start:fig5]
    # -------------------------------------------------------------
    # Figura 5 — raios ||x|| calculados em 5D
    # -------------------------------------------------------------

    radius_a = np.linalg.norm(
        class_a,
        axis=1,
    )

    radius_b = np.linalg.norm(
        class_b,
        axis=1,
    )

    radius_c = np.linalg.norm(
        class_c,
        axis=1,
    )

    radius_d = np.linalg.norm(
        class_d,
        axis=1,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].hist(
        radius_a,
        bins=30,
        alpha=0.55,
        label="Classe A",
    )

    axes[0].hist(
        radius_b,
        bins=30,
        alpha=0.55,
        label="Classe B",
    )

    axes[0].set_title(
        "Dataset I — distribuição de $||x||$"
    )
    axes[0].set_xlabel("Raio $||x||$")
    axes[0].set_ylabel("Frequência")
    axes[0].legend()

    axes[1].hist(
        radius_c,
        bins=30,
        alpha=0.55,
        label="Classe C",
    )

    axes[1].hist(
        radius_d,
        bins=30,
        alpha=0.55,
        label="Classe D",
    )

    axes[1].set_title(
        "Dataset II — distribuição de $||x||$"
    )
    axes[1].set_xlabel("Raio $||x||$")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()

    fig.suptitle(
        "Figura 5 — Distribuição dos raios em 5D"
    )

    fig.tight_layout()

    save(
        fig,
        "fig05-radius.png",
    )
    # --8<-- [end:fig5]

    # --8<-- [start:radius-rule]
    # -------------------------------------------------------------
    # Uma funcao nao linear simples para Dataset II.
    #
    # Centro entre os dois raios:
    # (2 + 5) / 2 = 3.5.
    #
    # Logo:
    # ||x||^2 > 3.5^2 => classe D
    # -------------------------------------------------------------

    threshold_sq = 3.5 ** 2

    shell_prediction = (
        np.sum(X2 ** 2, axis=1)
        > threshold_sq
    ).astype(int)

    shell_errors = int(
        np.sum(
            shell_prediction != y2
        )
    )
    # --8<-- [end:radius-rule]

    # -------------------------------------------------------------
    # Resultados.
    # -------------------------------------------------------------

    print(
        f"Dataset I - distância entre centros: "
        f"{distance_1:.3f}"
    )

    print(
        f"Dataset II - distância entre centros: "
        f"{distance_2:.3f}"
    )

    print(
        f"Dataset I - variância explicada PC1+PC2: "
        f"{explained_1:.2%}"
    )

    print(
        f"Dataset II - variância explicada PC1+PC2: "
        f"{explained_2:.2%}"
    )

    print(
        f"Raio médio Classe C: "
        f"{radius_c.mean():.3f}"
    )

    print(
        f"Raio médio Classe D: "
        f"{radius_d.mean():.3f}"
    )

    print(
        f"Regra sum(x_i^2) > {threshold_sq:.2f}: "
        f"{shell_errors} erros em {len(y2)} pontos"
    )


if __name__ == "__main__":
    main()