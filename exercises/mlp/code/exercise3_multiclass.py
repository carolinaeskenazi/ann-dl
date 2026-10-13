"""Exercise 3 - Multi-Class, Same Network.

Treina o MESMO ``MLP`` do Exercicio 2 em um problema de 3 classes. So
mudam os argumentos: tamanho das camadas e o par saida/perda
(``SoftmaxCCE`` no lugar de ``SigmoidBCE``). Salva as Figuras 4 e 5.

Uso, a partir da raiz do repositorio:

    python docs/exercises/mlp/code/exercise3_multiclass.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from datasets import load_multiclass, one_hot, split
from mlp import MLP, SoftmaxCCE
from plots import COLORS, plot_loss, save


# --8<-- [start:config]
N_CLASSES = 3

LAYERS = [4, 16, N_CLASSES]  # 4 entradas, 1 camada oculta de 16 neuronios tanh, 3 saidas
ETA = 0.1
EPOCHS = 2000
# --8<-- [end:config]


# --8<-- [start:run]
def run(layers: list[int], seed: int, data) -> dict:
    """Treina uma rede e devolve tudo o que o relatorio usa.

    O Exercicio 4 chama esta mesma funcao, com os mesmos ETA e EPOCHS,
    trocando apenas ``layers`` -- e assim que a comparacao fica controlada.
    """

    X_train, X_test, y_train, y_test = data

    model = MLP(layers, head=SoftmaxCCE, rng=np.random.default_rng(seed), learning_rate=ETA)

    Y_train = one_hot(y_train, N_CLASSES)
    history = model.fit(X_train, Y_train, epochs=EPOCHS)

    return {
        "model": model,
        "history": history,
        "train_loss": model.loss(X_train, Y_train),
        "train_acc": model.accuracy(X_train, y_train),
        "test_acc": model.accuracy(X_test, y_test),
    }
# --8<-- [end:run]


def main() -> None:
    # --8<-- [start:data]
    # -----------------------------------------------------------------
    # Item A - dados e divisao
    # -----------------------------------------------------------------

    X, y = load_multiclass()
    data = split(X, y)
    X_train, X_test, y_train, y_test = data
    # --8<-- [end:data]

    # --8<-- [start:train]
    # -----------------------------------------------------------------
    # Item C - treino e avaliacao
    # -----------------------------------------------------------------

    result = run(LAYERS, seed=42, data=data)
    model = result["model"]

    cm = confusion_matrix(y_test, model.predict(X_test))

    # Baseline do item D: mesma divisao, sem camada oculta.
    baseline = LogisticRegression().fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    baseline_cm = confusion_matrix(y_test, baseline.predict(X_test))
    # --8<-- [end:train]

    # --8<-- [start:fig4]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    plot_loss(ax, result["history"], label=f"MLP {LAYERS}, $\\eta$ = {ETA}", color=COLORS[0], linewidth=2.0)

    ax.set_ylabel("Perda de treino (entropia cruzada categórica)")
    ax.set_title("Perda de treino por época — Exercise 3", pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig04-loss.png")
    # --8<-- [end:fig4]

    # --8<-- [start:fig5]
    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    image = ax.imshow(cm, cmap="Blues")

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "#111111",
            )

    ticks = [f"Classe {c}" for c in range(N_CLASSES)]
    ax.set_xticks(range(N_CLASSES), ticks)
    ax.set_yticks(range(N_CLASSES), ticks)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Matriz de confusão — teste (acc = {result['test_acc']:.3f})", pad=10)

    fig.colorbar(image, ax=ax, label="Número de amostras")

    fig.tight_layout()
    save(fig, "fig05-confusion.png")
    # --8<-- [end:fig5]

    # -----------------------------------------------------------------
    # Numeros do relatorio
    # -----------------------------------------------------------------

    print("EXERCICIO 3 - MULTICLASSE\n")

    print(f"treino: {len(y_train)} | teste: {len(y_test)}")
    print(f"amostras por classe no teste: {np.bincount(y_test)}\n")

    print(f"arquitetura: {LAYERS} ({model.n_parameters} parametros)")
    print(f"eta = {ETA} | epocas = {EPOCHS}")
    print(f"perda de treino: epoca 1 = {result['history'][0]:.4f} | final = {result['train_loss']:.4f}")
    print(f"acuracia de treino: {result['train_acc']:.4f}")
    print(f"acuracia de teste : {result['test_acc']:.4f}\n")

    print("matriz de confusao (linha = verdadeira, coluna = prevista):")
    print(cm)

    print("\nerros por par de classes (i<->j = cm[i,j] + cm[j,i]):")
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            print(f"  {i}<->{j}: {cm[i, j] + cm[j, i]}  ({i}->{j}: {cm[i, j]}, {j}->{i}: {cm[j, i]})")

    print(f"\nacerto por classe (recall): {np.round(cm.diagonal() / cm.sum(axis=1), 4)}")

    print(f"\nbaseline (regressao logistica): acuracia de teste {baseline_acc:.4f}")
    print(baseline_cm)
    print(f"acerto por classe da baseline: {np.round(baseline_cm.diagonal() / baseline_cm.sum(axis=1), 4)}")


if __name__ == "__main__":
    main()
