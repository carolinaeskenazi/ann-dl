"""Exercise 2 - A 2-D latent space.

Treina o VAE com d = 2, reaproveita o modelo d = 16 do Exercicio 1 (ou o
treina, se ``.cache/vae/vae_d16.pt`` nao existir), salva as Figuras 4, 5 e 6
em ``figures/`` e imprime os numeros usados no relatorio.

Uso, a partir da raiz do repositorio:

    python docs/exercises/vae/code/exercise2_latent2d.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from data import load_mnist
from plots import clean_axes, save, show_digits, tile
from vae import evaluate, load_or_train, save_model, train


# --8<-- [start:config]
LATENT_DIM = 2

# Item C: 15 quantis igualmente espacados, levados ao eixo de z por Phi^-1.
GRID = 15
QUANTILES = np.linspace(0.05, 0.95, GRID)

# Item D: duas imagens de teste de digitos diferentes (um 0 e um 1).
IDX_A, IDX_B = 3, 2
STEPS = 10
# --8<-- [end:config]


def main() -> None:
    # --8<-- [start:train]
    # -----------------------------------------------------------------
    # Item A - mesmo modelo, semente, beta e orcamento; so d muda
    # -----------------------------------------------------------------

    train_set, val_set, test_set = load_mnist()
    X_test, y_test = test_set.tensors

    model2, history2 = train(LATENT_DIM, train_set, val_set)
    save_model(model2, history2)

    model16, _ = load_or_train(16, train_set, val_set)

    results = {d: evaluate(m, test_set) for d, m in [(2, model2), (16, model16)]}
    # --8<-- [end:train]

    # --8<-- [start:fig4]
    # -----------------------------------------------------------------
    # Item B - mu das 10 000 imagens de teste
    # -----------------------------------------------------------------

    model2.eval()
    with torch.no_grad():
        mu_test, _ = model2.encode(X_test)

    mu_test = mu_test.numpy()
    labels = y_test.numpy()
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    for digit in range(10):
        pts = mu_test[labels == digit]
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, color=colors(digit), label=str(digit), rasterized=True)

    # Quadrado coberto pela grade da Figura 5: Phi^-1(0.05) a Phi^-1(0.95).
    edge = norm.ppf(0.95)
    ax.add_patch(plt.Rectangle((-edge, -edge), 2 * edge, 2 * edge, fill=False, ls="--", lw=1.2, color="#333333"))

    ax.set_aspect("equal")
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("$\\mu$ das 10 000 imagens de teste — d = 2", pad=10)
    ax.legend(title="Dígito", markerscale=4, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    clean_axes(ax)

    fig.tight_layout()
    save(fig, "fig04-latent.png")
    # --8<-- [end:fig4]

    # --8<-- [start:fig5]
    # -----------------------------------------------------------------
    # Item C - o plano decodificado
    # -----------------------------------------------------------------

    z_axis = norm.ppf(QUANTILES)

    # Linha r da grade = z2 decrescente (como no eixo y da Figura 4);
    # coluna c = z1 crescente.
    z1, z2 = np.meshgrid(z_axis, z_axis[::-1])
    z_grid = torch.tensor(np.column_stack([z1.ravel(), z2.ravel()]), dtype=torch.float32)

    with torch.no_grad():
        plane = model2.decode(z_grid)

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    show_digits(ax, tile(plane, rows=GRID, cols=GRID))

    centers = 2 + 30 * np.arange(GRID) + 14  # centro de cada celula em pixels
    ax.set_xticks(centers[::2])
    ax.set_xticklabels([f"{v:.2f}" for v in z_axis[::2]], fontsize=9)
    ax.set_yticks(centers[::2])
    ax.set_yticklabels([f"{v:.2f}" for v in z_axis[::-1][::2]], fontsize=9)
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
    ax.set_title("Plano latente decodificado, grade 15 × 15 em $\\Phi^{-1}$ — d = 2", pad=10)

    fig.tight_layout()
    save(fig, "fig05-plane.png")
    # --8<-- [end:fig5]

    # --8<-- [start:fig6]
    # -----------------------------------------------------------------
    # Item D - interpolacao no espaco de pixels x no espaco latente (d = 16)
    # -----------------------------------------------------------------

    x_a, x_b = X_test[IDX_A], X_test[IDX_B]
    t = torch.linspace(0, 1, STEPS).unsqueeze(1)

    pixel_path = (1 - t) * x_a + t * x_b

    model16.eval()
    with torch.no_grad():
        mu_ab, _ = model16.encode(torch.stack([x_a, x_b]))
        latent_path = model16.decode((1 - t) * mu_ab[0] + t * mu_ab[1])

    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    show_digits(ax, tile(torch.cat([pixel_path, latent_path]), rows=2, cols=STEPS))

    ax.set_yticks([15, 45])
    ax.set_yticklabels(["Pixels", "Latente (d = 16)"])
    ax.set_xticks(2 + 30 * np.arange(STEPS) + 14)
    ax.set_xticklabels([f"{v:.2f}" for v in t.squeeze().tolist()], fontsize=9)
    ax.tick_params(length=0)
    ax.set_xlabel("t")
    ax.set_title(
        f"Interpolação entre as imagens de teste {IDX_A} (dígito {int(y_test[IDX_A])}) "
        f"e {IDX_B} (dígito {int(y_test[IDX_B])})",
        pad=8,
    )

    fig.tight_layout()
    save(fig, "fig06-interpolation.png")
    # --8<-- [end:fig6]

    # -----------------------------------------------------------------
    # Numeros do relatorio
    # -----------------------------------------------------------------

    print("\nEXERCICIO 2 - ESPACO LATENTE 2-D\n")

    print("--- teste (nats por imagem) ---")
    for d, (rec, kl_v) in results.items():
        print(f"d = {d:2d}: reconstrucao {rec:.2f} | KL {kl_v:.2f} | ELBO negativo {rec + kl_v:.2f}")

    print(f"\nKL de validacao d = 2: epoca 1 = {history2['val_kl'][1]:.2f} | "
          f"epoca 20 = {history2['val_kl'][20]:.2f}")

    print(f"\nquantis: {QUANTILES.round(4)}")
    print(f"z da grade: {z_axis.round(3)}")
    print(f"mu de teste dentro do quadrado da grade: "
          f"{np.mean(np.all(np.abs(mu_test) <= edge, axis=1)):.1%}")

    # Sobreposicao entre digitos no plano, medida sem o teste: k-NN ajustado
    # nos mu de treino e avaliado nos mu de validacao.
    with torch.no_grad():
        mu_train, _ = model2.encode(train_set.dataset.tensors[0][train_set.indices])
        mu_val, _ = model2.encode(val_set.dataset.tensors[0][val_set.indices])
    y_train = train_set.dataset.tensors[1][train_set.indices].numpy()
    y_val = val_set.dataset.tensors[1][val_set.indices].numpy()

    knn = KNeighborsClassifier(n_neighbors=15).fit(mu_train.numpy(), y_train)
    cm = confusion_matrix(y_val, knn.predict(mu_val.numpy()))
    print(f"\nk-NN (k = 15) no plano, acuracia de validacao: {np.trace(cm) / cm.sum():.4f}")

    pairs = sorted(
        ((cm[i, j] + cm[j, i], i, j) for i in range(10) for j in range(i + 1, 10)),
        reverse=True,
    )
    print("pares mais confundidos (validacao):")
    for total, i, j in pairs[:6]:
        print(f"  {i} <-> {j}: {total} ({i}->{j}: {cm[i, j]}, {j}->{i}: {cm[j, i]})")

    # Que digito o k-NN atribui a cada celula da Figura 5 (linha de cima = z2 alto).
    cell_digit = knn.predict(z_grid.numpy()).reshape(GRID, GRID)
    print("\ndigito do vizinho por celula da Figura 5:")
    for row in cell_digit:
        print("  " + " ".join(str(v) for v in row))

    print(f"\ninterpolacao: teste {IDX_A} (digito {int(y_test[IDX_A])}) -> "
          f"teste {IDX_B} (digito {int(y_test[IDX_B])})")


if __name__ == "__main__":
    main()
