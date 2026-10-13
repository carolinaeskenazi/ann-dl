"""Exercise 4 - Does a Second Hidden Layer Help?

Mesmos dados, mesma divisao, mesmo eta e mesmo numero de epocas do
Exercicio 3 -- importados de ``exercise3_multiclass.py`` para que nao
possam divergir. A unica mudanca e a profundidade. Salva a Figura 6.

Uso, a partir da raiz do repositorio:

    python docs/exercises/mlp/code/exercise4_depth.py
"""

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_multiclass, split
from exercise3_multiclass import EPOCHS, ETA, LAYERS as SHALLOW, run
from plots import COLORS, plot_loss, save


# --8<-- [start:config]
# Uma segunda camada oculta com a mesma largura da primeira.
DEEP = [4, 16, 16, 3]

# Controle: 1 camada oculta, mas com quase o mesmo numero de parametros
# da rede profunda (387 contra 403). Separa "mais fundo" de "mais parametros".
WIDE = [4, 48, 3]

# A semente 42 e a rodada principal; as outras entram na media do item B.
SEEDS = [42, 43, 44, 45, 46]
# --8<-- [end:config]


def main() -> None:
    # --8<-- [start:train]
    data = split(*load_multiclass())

    # Cada semente inicializa as duas redes; os dados e a divisao nao mudam.
    shallow = [run(SHALLOW, seed, data) for seed in SEEDS]
    deep = [run(DEEP, seed, data) for seed in SEEDS]
    wide = [run(WIDE, seed, data) for seed in SEEDS]
    # --8<-- [end:train]

    # --8<-- [start:fig6]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for results, layers, color, name in (
        (shallow, SHALLOW, COLORS[0], "Exercise 3"),
        (deep, DEEP, COLORS[1], "Exercise 4"),
    ):
        # Linha grossa: semente 42. Linhas finas: as outras sementes.
        for r in results[1:]:
            ax.plot(range(1, EPOCHS + 1), r["history"], color=color, linewidth=0.8, alpha=0.35)

        plot_loss(ax, results[0]["history"], label=f"{name} — {layers}", color=color, linewidth=2.2)

    ax.set_ylabel("Perda de treino (entropia cruzada categórica)")
    ax.set_title(f"Perda de treino — 1 vs. 2 camadas ocultas ({len(SEEDS)} sementes)", pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig06-depth.png")
    # --8<-- [end:fig6]

    # -----------------------------------------------------------------
    # Numeros do relatorio
    # -----------------------------------------------------------------

    print("EXERCICIO 4 - PROFUNDIDADE\n")
    print(f"eta = {ETA} | epocas = {EPOCHS} | sementes = {SEEDS}\n")

    for results, layers in ((shallow, SHALLOW), (deep, DEEP), (wide, WIDE)):
        loss = np.array([r["train_loss"] for r in results])
        train = np.array([r["train_acc"] for r in results])
        test = np.array([r["test_acc"] for r in results])

        print(f"--- {layers} ({results[0]['model'].n_parameters} parametros) ---")
        print(f"semente 42: perda final {loss[0]:.4f} | treino {train[0]:.4f} | teste {test[0]:.4f}")
        print(f"perda final por semente: {np.round(loss, 4)}")
        print(f"teste por semente      : {np.round(test, 4)}")
        print(f"media: perda {loss.mean():.4f} | treino {train.mean():.4f} | teste {test.mean():.4f} (desvio {test.std(ddof=1):.4f})\n")

    diff = np.array([d["test_acc"] - s["test_acc"] for d, s in zip(deep, shallow)])
    print(f"diferenca de teste (Ex. 4 - Ex. 3) por semente: {np.round(diff, 4)}")
    print(f"diferenca media: {diff.mean():+.4f}")

    # Quanto de perda de treino vira acerto de teste.
    gap_s = np.mean([r["train_acc"] - r["test_acc"] for r in shallow])
    gap_d = np.mean([r["train_acc"] - r["test_acc"] for r in deep])
    print(f"folga treino - teste (media): Ex. 3 {gap_s:.4f} | Ex. 4 {gap_d:.4f}")


if __name__ == "__main__":
    main()
