"""Exercise 2 - Binary Classification a Straight Line Cannot Do.

Gera os dados do item A, mede a baseline linear, treina o MLP de
``mlp.py``, salva as Figuras 1, 2 e 3 em ``figures/`` e imprime os
numeros usados no relatorio.

Uso, a partir da raiz do repositorio:

    python docs/exercises/mlp/code/exercise2_binary.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

from datasets import load_blobs, split
from mlp import MLP, SigmoidBCE
from plots import COLORS, plot_loss, save, scatter_classes


# --8<-- [start:config]
rng = np.random.default_rng(42)

LAYERS = [2, 8, 1]  # 2 entradas, 1 camada oculta de 8 neuronios tanh, 1 saida
ETA = 0.1
EPOCHS = 2000
# --8<-- [end:config]


def main() -> None:
    # --8<-- [start:data]
    # -----------------------------------------------------------------
    # Item A - dados, divisao e Figura 1
    # -----------------------------------------------------------------

    X, y = load_blobs()
    X_train, X_test, y_train, y_test = split(X, y)
    # --8<-- [end:data]

    # --8<-- [start:fig1]
    fig, ax = plt.subplots(figsize=(6.6, 6.0))

    scatter_classes(ax, X, y)

    ax.set_title("Os 1000 pontos — classe 1 dos dois lados da classe 0", pad=10)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig01-data.png")
    # --8<-- [end:fig1]

    # --8<-- [start:baseline]
    # -----------------------------------------------------------------
    # Item B - baseline: uma unica fronteira linear
    # -----------------------------------------------------------------

    baseline = LogisticRegression().fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)

    # Referencia para o item D: a melhor reta perpendicular a diagonal,
    # x1 + x2 = t, com t escolhido no treino. Ela isola um dos dois
    # clusters da classe 1.
    s_train = X_train.sum(axis=1)
    thresholds = np.sort(s_train)
    train_accs = [np.mean((s_train >= t).astype(int) == y_train) for t in thresholds]
    t_best = thresholds[int(np.argmax(train_accs))]
    line_acc = float(np.mean((X_test.sum(axis=1) >= t_best).astype(int) == y_test))
    # --8<-- [end:baseline]

    # --8<-- [start:train]
    # -----------------------------------------------------------------
    # Item C - MLP
    # -----------------------------------------------------------------

    model = MLP(LAYERS, head=SigmoidBCE, rng=rng, learning_rate=ETA)

    # A saida tem 1 neuronio: os rotulos viram uma coluna (N, 1).
    history = model.fit(X_train, y_train.reshape(-1, 1).astype(float), epochs=EPOCHS)

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    # --8<-- [end:train]

    # --8<-- [start:fig2]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    plot_loss(ax, history, label=f"MLP {LAYERS}, $\\eta$ = {ETA}", color=COLORS[0], linewidth=2.0)

    ax.set_ylabel("Perda de treino (entropia cruzada binária)")
    ax.set_title("Perda de treino por época — Exercise 2", pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig02-loss.png")
    # --8<-- [end:fig2]

    # --8<-- [start:fig3]
    # -----------------------------------------------------------------
    # Figura 3 - fronteira de decisao sobre os pontos de teste
    # -----------------------------------------------------------------

    # A rede e avaliada em uma grade que cobre o plano; cada no da grade
    # recebe a classe prevista, e contourf pinta as duas regioes.
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400)
    x2 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400)
    G1, G2 = np.meshgrid(x1, x2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])

    regions = model.predict(grid).reshape(G1.shape)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))

    ax.contourf(
        G1,
        G2,
        regions,
        levels=[-0.5, 0.5, 1.5],
        cmap=ListedColormap([COLORS[0], COLORS[1]]),
        alpha=0.18,
    )

    scatter_classes(ax, X_test, y_test, alpha=0.9)

    # Fronteira da baseline: w . x + b = 0.
    (w1, w2), b = baseline.coef_[0], baseline.intercept_[0]
    ax.plot(
        x1,
        -(w1 * x1 + b) / w2,
        color="#111111",
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline linear (acc = {baseline_acc:.3f})",
    )

    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_aspect("equal")
    ax.set_title(f"Fronteira de decisão do MLP — teste (acc = {test_acc:.3f})", pad=10)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig03-boundary.png")
    # --8<-- [end:fig3]

    # -----------------------------------------------------------------
    # Numeros do relatorio
    # -----------------------------------------------------------------

    print("EXERCICIO 2 - CLASSIFICACAO BINARIA\n")

    print(f"treino: {len(y_train)} pontos | teste: {len(y_test)} pontos")
    print(f"classe 1 no treino: {y_train.mean():.3f} | no teste: {y_test.mean():.3f}\n")

    print("--- baseline (regressao logistica) ---")
    print(f"w = [{w1:.6f}, {w2:.6f}] | b = {b:.6f}")
    print(f"acuracia de teste: {baseline_acc:.4f}")
    print(f"previsoes no teste: {int(baseline.predict(X_test).sum())} pontos na classe 1")
    print(f"reta x1 + x2 = {t_best:.4f}: acuracia de teste {line_acc:.4f}\n")

    print("--- MLP ---")
    print(f"arquitetura: {LAYERS} ({model.n_parameters} parametros)")
    print(f"eta = {ETA} | epocas = {EPOCHS}")
    print(f"perda de treino: epoca 1 = {history[0]:.4f} | final = {model.loss(X_train, y_train.reshape(-1, 1)):.4f}")
    print(f"acuracia de treino: {train_acc:.4f}")
    print(f"acuracia de teste : {test_acc:.4f} ({int(round((1 - test_acc) * len(y_test)))} erros em {len(y_test)})")

    # Em quais regioes da diagonal a rede preve classe 1.
    diag = np.linspace(-7, 7, 1401)
    pred = model.predict(np.column_stack([diag, diag]))
    cuts = diag[1:][np.diff(pred) != 0]
    print(f"trocas de classe ao longo da diagonal x1 = x2: {np.round(cuts, 2)}")


if __name__ == "__main__":
    main()
