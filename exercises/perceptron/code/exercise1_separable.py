"""Exercise 1 - Separable Data: the case the perceptron was designed for.

Gera duas gaussianas 2D linearmente separaveis, treina o perceptron de
``perceptron.py``, salva as Figuras 1, 2 e 3 em ``figures/`` e imprime os
numeros usados no relatorio:

- w, b, epocas e acuracia finais com eta = 0.01;
- a mesma rodada com eta = 1.0, para comparar direcoes de w;
- o numero de atualizacoes por epoca.

Uso, a partir da raiz do repositorio:

    python docs/exercises/perceptron/code/exercise1_separable.py
"""

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from plots import COLORS, mark_errors, plot_boundary, save, scatter_classes


# ---------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------

# --8<-- [start:config]
# Semente fixa pedida pelo enunciado; o mesmo rng e usado no script todo.
rng = np.random.default_rng(42)

N_PER_CLASS = 1000
MAX_EPOCHS = 100

# Classe 0 e classe 1: medias distantes em relacao a dispersao.
MEAN_0 = np.array([1.5, 1.5])
MEAN_1 = np.array([5.0, 5.0])
COV = np.array([[0.5, 0.0], [0.0, 0.5]])

ETA = 0.01        # taxa de aprendizado principal
ETA_LARGE = 1.0   # rodada de comparacao do item D
# --8<-- [end:config]


# ---------------------------------------------------------------------
# Geracao dos dados
# ---------------------------------------------------------------------

# --8<-- [start:generate]
def generate(
    mean_0: np.ndarray,
    mean_1: np.ndarray,
    cov: np.ndarray,
    n_per_class: int = N_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera n pontos por classe a partir de duas normais multivariadas.

    As amostras ficam empilhadas na ordem natural (classe 0 e depois
    classe 1); o laco de treino percorre o conjunto nessa mesma ordem.
    """

    X_0 = rng.multivariate_normal(mean_0, cov, size=n_per_class)
    X_1 = rng.multivariate_normal(mean_1, cov, size=n_per_class)

    X = np.vstack([X_0, X_1])

    y = np.concatenate(
        [
            np.zeros(n_per_class, dtype=int),
            np.ones(n_per_class, dtype=int),
        ]
    )

    return X, y
# --8<-- [end:generate]


def direction(w: np.ndarray) -> np.ndarray:
    """Vetor unitario na direcao de w (a direcao normal a fronteira)."""

    return w / np.linalg.norm(w)


def angle_deg(w: np.ndarray) -> float:
    """Angulo de w com o eixo x1, em graus."""

    return float(np.degrees(np.arctan2(w[1], w[0])))


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------

def main() -> None:
    # --8<-- [start:data]
    # -----------------------------------------------------------------
    # Item A - dados e Figura 1
    # -----------------------------------------------------------------

    X, y = generate(MEAN_0, MEAN_1, COV)

    # Inicializacao pedida no item B: w pequeno e nao nulo, b = 0.
    # E sorteado uma unica vez e reaproveitado pelas duas taxas de
    # aprendizado, para que a unica diferenca entre as rodadas seja eta.
    w0 = rng.normal(0, 0.01, size=2)
    b0 = 0.0
    # --8<-- [end:data]

    # --8<-- [start:fig1]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    scatter_classes(ax, X, y)

    ax.set_title(
        "Dados separáveis — 1000 pontos por classe",
        pad=10,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    save(
        fig,
        "fig01-separable-data.png",
    )
    # --8<-- [end:fig1]

    # --8<-- [start:train]
    # -----------------------------------------------------------------
    # Itens B e C - treino com eta = 0.01
    # -----------------------------------------------------------------

    model = Perceptron(w0=w0, b0=b0, learning_rate=ETA)
    result = model.fit(X, y, max_epochs=MAX_EPOCHS)

    # Item D - mesma coisa, mudando apenas eta (mesmo w0, mesmo b0).
    model_large = Perceptron(w0=w0, b0=b0, learning_rate=ETA_LARGE)
    result_large = model_large.fit(X, y, max_epochs=MAX_EPOCHS)
    # --8<-- [end:train]

    y_pred = model.predict(X)

    # --8<-- [start:fig2]
    # -----------------------------------------------------------------
    # Figura 2 - fronteira de decisao e pontos classificados errado
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    scatter_classes(ax, X, y)

    plot_boundary(
        ax,
        result.w,
        result.b,
        label=r"Fronteira $w \cdot x + b = 0$",
        color="#111111",
        linestyle="-",
        linewidth=2.5,
    )

    n_wrong = mark_errors(
        ax,
        X,
        y,
        y_pred,
        label="Classificados errado",
    )

    ax.set_title(
        f"Fronteira aprendida — $\eta$ = {ETA}, "
        f"{result.epochs} épocas",
        pad=10,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    save(
        fig,
        "fig02-boundary.png",
    )
    # --8<-- [end:fig2]

    # --8<-- [start:fig3]
    # -----------------------------------------------------------------
    # Figura 3 - acuracia por epoca (as duas taxas de aprendizado)
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        range(1, result.epochs + 1),
        result.accuracy_per_epoch,
        color=COLORS[0],
        linewidth=2.2,
        marker="o",
        markersize=4,
        label=rf"$\eta = {ETA}$",
    )

    ax.plot(
        range(1, result_large.epochs + 1),
        result_large.accuracy_per_epoch,
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
        label=rf"$\eta = {ETA_LARGE}$",
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no conjunto completo")

    ax.set_title(
        "Acurácia por época — dados separáveis",
        pad=10,
    )

    ax.set_ylim(0.45, 1.05)

    ax.set_xticks(
        range(
            0,
            max(result.epochs, result_large.epochs) + 1,
            5,
        )
    )

    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    save(
        fig,
        "fig03-accuracy.png",
    )
    
    # --8<-- [end:fig3]

    # --8<-- [start:report]
    # -----------------------------------------------------------------
    # Numeros do relatorio
    # -----------------------------------------------------------------

    print("EXERCICIO 1 - DADOS SEPARAVEIS\n")

    print(f"w0 inicial : [{w0[0]:.6f}, {w0[1]:.6f}]  |  b0 = {b0:.1f}")
    print(f"||w0||     : {np.linalg.norm(w0):.6f}")
    print(f"||x|| medio: {np.linalg.norm(X, axis=1).mean():.6f}")

    # Quanto um unico erro move w: e a escala com que ||w0|| compete.
    for eta in (ETA, ETA_LARGE):
        print(
            f"passo de uma atualizacao com eta = {eta}: "
            f"eta * ||x|| = {eta * np.linalg.norm(X, axis=1).mean():.6f}"
        )

    print()

    for name, res in (
        (f"eta = {ETA}", result),
        (f"eta = {ETA_LARGE}", result_large),
    ):
        u = direction(res.w)

        print(f"--- {name} ---")
        print(f"w final      : [{res.w[0]:.6f}, {res.w[1]:.6f}]")
        print(f"b final      : {res.b:.6f}")
        print(f"||w||        : {np.linalg.norm(res.w):.6f}")
        print(f"w / ||w||    : [{u[0]:.6f}, {u[1]:.6f}]")
        print(f"angulo de w  : {angle_deg(res.w):.3f} graus")

        # Distancia da fronteira a origem, medida ao longo de w. Como as
        # duas direcoes sao quase iguais, e este numero que distingue as
        # duas retas.
        print(f"|b| / ||w||  : {abs(res.b) / np.linalg.norm(res.w):.4f}")
        print(
            f"centros proj.: {MEAN_0 @ u:.4f} (classe 0) e "
            f"{MEAN_1 @ u:.4f} (classe 1)"
        )
        print(f"epocas       : {res.epochs} (convergiu: {res.converged})")
        print(f"acuracia     : {res.accuracy:.4f}")
        n_err = int(round((1 - res.accuracy) * len(y)))
        print(f"erros        : {n_err} de {len(y)} pontos")
        print(f"atualizacoes por epoca: {res.updates_per_epoch}\n")

    cos = float(
        np.clip(direction(result.w) @ direction(result_large.w), -1.0, 1.0)
    )

    print(
        f"Angulo entre as duas direcoes de w: "
        f"{np.degrees(np.arccos(cos)):.3f} graus"
    )

    print(f"Pontos classificados errado na Figura 2: {n_wrong}")
    # --8<-- [end:report]


if __name__ == "__main__":
    main()
