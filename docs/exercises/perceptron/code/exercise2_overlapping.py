"""Exercise 2 - Overlapping Data: the case the perceptron cannot solve.

Gera duas gaussianas 2D que se sobrepoem fortemente, treina o MESMO
perceptron de ``perceptron.py`` (sem alteracao, apenas com o
rastreamento do bolso ligado), salva as Figuras 4, 5 e 6 em ``figures/``
e imprime os numeros usados no relatorio.

Uso, a partir da raiz do repositorio:

    python docs/exercises/perceptron/code/exercise2_overlapping.py
"""

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from plots import (
    COLORS,
    FINAL_STYLE,
    POCKET_STYLE,
    mark_errors,
    plot_boundary,
    save,
    scatter_classes,
)


# ---------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------

# --8<-- [start:config]
# Mesma semente e mesmo protocolo do Exercicio 1.
rng = np.random.default_rng(42)

N_PER_CLASS = 1000
MAX_EPOCHS = 100
ETA = 0.01

# Medias proximas e dispersao tres vezes maior: as nuvens se misturam.
MEAN_0 = np.array([3.0, 3.0])
MEAN_1 = np.array([4.0, 4.0])
COV = np.array([[1.5, 0.0], [0.0, 1.5]])
# --8<-- [end:config]


# --8<-- [start:generate]
def generate(
    mean_0: np.ndarray,
    mean_1: np.ndarray,
    cov: np.ndarray,
    n_per_class: int = N_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesma geracao do Exercicio 1, com os parametros do item A."""

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


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------

def main() -> None:
    # --8<-- [start:data]
    # -----------------------------------------------------------------
    # Item A - dados e Figura 4
    # -----------------------------------------------------------------

    X, y = generate(MEAN_0, MEAN_1, COV)

    w0 = rng.normal(0, 0.01, size=2)
    b0 = 0.0
    # --8<-- [end:data]

    # --8<-- [start:fig4]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    scatter_classes(ax, X, y)

    ax.set_title(
        "Dados sobrepostos — 1000 pontos por classe",
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
        "fig04-overlapping-data.png",
    )
    # --8<-- [end:fig4]

    # --8<-- [start:train]
    # -----------------------------------------------------------------
    # Item B - treino com o bolso ligado
    # -----------------------------------------------------------------

    model = Perceptron(w0=w0, b0=b0, learning_rate=ETA)
    result = model.fit(X, y, max_epochs=MAX_EPOCHS, pocket=True)

    # Dois conjuntos de pesos: o ultimo iterado e o melhor ja visto.
    final_pred = model.predict(X)

    pocket = Perceptron(w0=result.pocket_w, b0=result.pocket_b)
    pocket_pred = pocket.predict(X)
    # --8<-- [end:train]

    # --8<-- [start:fig5]
    # -----------------------------------------------------------------
    # Figura 5 - as duas fronteiras sobre os dados
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    scatter_classes(ax, X, y)

    n_wrong_pocket = mark_errors(
        ax,
        X,
        y,
        pocket_pred,
        label="Erros do bolso",
    )

    plot_boundary(
        ax,
        result.w,
        result.b,
        label=f"Fronteira final (acc = {result.accuracy:.3f})",
        **FINAL_STYLE,
    )

    plot_boundary(
        ax,
        result.pocket_w,
        result.pocket_b,
        label=(
            f"Fronteira do bolso "
            f"(acc = {result.pocket_accuracy:.3f})"
        ),
        **POCKET_STYLE,
    )

    ax.set_title(
        "Fronteiras final e do bolso — dados sobrepostos",
        pad=10,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    save(
        fig,
        "fig05-boundaries.png",
    )
    # --8<-- [end:fig5]

    # --8<-- [start:fig6]
    # -----------------------------------------------------------------
    # Figura 6 - acuracia atual e melhor-ate-agora por epoca
    # -----------------------------------------------------------------

    epochs = range(1, result.epochs + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        epochs,
        result.accuracy_per_epoch,
        color=COLORS[0],
        linewidth=2.0,
        label="Acurácia dos pesos atuais",
    )

    ax.plot(
        epochs,
        result.pocket_accuracy_per_epoch,
        color=POCKET_STYLE["color"],
        linestyle="-",
        linewidth=2.5,
        label="Melhor-até-agora (bolso)",
    )

    ax.axhline(
        result.pocket_accuracy,
        color=POCKET_STYLE["color"],
        linestyle=":",
        linewidth=1.4,
        alpha=0.7,
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no conjunto completo")

    ax.set_title(
        "Acurácia por época — dados sobrepostos",
        pad=10,
    )

    ax.set_ylim(0.45, 0.80)

    ax.set_xticks(
        range(
            0,
            result.epochs + 1,
            10,
        )
    )

    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="center right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    save(
        fig,
        "fig06-accuracy.png",
    )
    # --8<-- [end:fig6]

    # --8<-- [start:report]
    # -----------------------------------------------------------------
    # Numeros do relatorio
    # -----------------------------------------------------------------

    print("EXERCICIO 2 - DADOS SOBREPOSTOS\n")

    print(f"w0 inicial: [{w0[0]:.6f}, {w0[1]:.6f}]  |  b0 = {b0:.1f}\n")

    print("--- pesos finais (ultimo iterado) ---")
    print(f"w        : [{result.w[0]:.6f}, {result.w[1]:.6f}]")
    print(f"b        : {result.b:.6f}")
    print(f"acuracia : {result.accuracy:.4f}")
    print(
        f"previsoes: {int(np.sum(final_pred == 1))} pontos na classe 1, "
        f"{int(np.sum(final_pred == 0))} na classe 0"
    )

    print("\n--- pesos do bolso (melhor-ate-agora) ---")
    print(f"w        : [{result.pocket_w[0]:.6f}, {result.pocket_w[1]:.6f}]")
    print(f"b        : {result.pocket_b:.6f}")
    print(f"acuracia : {result.pocket_accuracy:.4f}")
    print(f"epoca    : {result.pocket_epoch}")
    print(f"erros    : {n_wrong_pocket} de {len(y)} pontos")

    print(f"\nepocas executadas: {result.epochs} (parou sozinho: {result.converged})")
    print(f"atualizacoes nas 5 primeiras epocas: {result.updates_per_epoch[:5]}")
    print(f"atualizacoes nas 5 ultimas epocas  : {result.updates_per_epoch[-5:]}")

    acc = np.array(result.accuracy_per_epoch)
    print(
        f"acuracia por epoca: minimo {acc.min():.4f}, "
        f"maximo {acc.max():.4f}, media {acc.mean():.4f}"
    )

    # Escala tipica dos dados, usada na analise do item D.
    print(f"\n||x|| medio: {np.linalg.norm(X, axis=1).mean():.4f}")
    # --8<-- [end:report]


if __name__ == "__main__":
    main()
