"""Exercise 1 - A VAE on MNIST.

Treina o VAE com d = 16, salva as Figuras 1, 2 e 3 em ``figures/``, guarda o
modelo em ``.cache/vae/`` (o Exercicio 2 reaproveita) e imprime os numeros
usados no relatorio.

Uso, a partir da raiz do repositorio:

    python docs/exercises/vae/code/exercise1_mnist.py
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import load_mnist
from plots import COLORS, clean_axes, save, show_digits, tile
from vae import EPOCHS, bce, evaluate, save_model, train


# --8<-- [start:config]
LATENT_DIM = 16
# --8<-- [end:config]


def main() -> None:
    # --8<-- [start:data]
    # -----------------------------------------------------------------
    # Item A - dados
    # -----------------------------------------------------------------

    train_set, val_set, test_set = load_mnist()
    X_test, y_test = test_set.tensors
    # --8<-- [end:data]

    # --8<-- [start:train]
    # -----------------------------------------------------------------
    # Item C - treino; todas as decisoes olham so para a validacao
    # -----------------------------------------------------------------

    model, history = train(LATENT_DIM, train_set, val_set)
    save_model(model, history)

    # O teste entra aqui pela primeira vez, depois do treino terminado.
    test_recon, test_kl = evaluate(model, test_set)
    # --8<-- [end:train]

    # --8<-- [start:fig1]
    epochs = range(1, EPOCHS + 1)

    fig, (ax_rec, ax_kl) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for ax, key, title in [
        (ax_rec, "recon", "Reconstrução (BCE)"),
        (ax_kl, "kl", "KL"),
    ]:
        ax.plot(epochs, history[f"train_{key}"], "o-", ms=3.5, color=COLORS["train"], label="Treino")
        # history["val_*"][0] e a epoca 0 (antes do treino): fica fora da curva.
        ax.plot(epochs, history[f"val_{key}"][1:], "o-", ms=3.5, color=COLORS["val"], label="Validação")

        ax.set_title(f"{title} por época — d = {LATENT_DIM}", pad=10)
        ax.set_xlabel("Época")
        ax.set_ylabel("nats por imagem")
        ax.set_xticks(range(0, EPOCHS + 1, 2))
        ax.legend(frameon=True, framealpha=0.95)
        clean_axes(ax)

    fig.tight_layout()
    save(fig, "fig01-loss.png")
    # --8<-- [end:fig1]

    # --8<-- [start:fig2]
    # -----------------------------------------------------------------
    # Item D - reconstrucoes (Figura 2) e amostras do prior (Figura 3)
    # -----------------------------------------------------------------

    model.eval()

    # Primeira imagem de teste de cada digito, na ordem 0, 1, ..., 9.
    first_idx = [int((y_test == digit).nonzero()[0]) for digit in range(10)]
    originals = X_test[first_idx]

    with torch.no_grad():
        mu, _ = model.encode(originals)
        reconstructions = model.decode(mu)  # decodifica a media, sem sortear z

    fig, ax = plt.subplots(figsize=(9.0, 2.3))
    show_digits(ax, tile(torch.cat([originals, reconstructions]), rows=2, cols=10))

    ax.set_yticks([15, 45])
    ax.set_yticklabels(["Original", "Reconstrução"])
    ax.tick_params(length=0)
    ax.set_xlabel("Primeira imagem de teste de cada dígito, de 0 a 9")
    ax.set_title(f"Reconstruções a partir de $\\mu$ — d = {LATENT_DIM}", pad=8)

    fig.tight_layout()
    save(fig, "fig02-reconstructions.png")
    # --8<-- [end:fig2]

    # --8<-- [start:fig3]
    torch.manual_seed(0)
    z = torch.randn(64, LATENT_DIM)  # z ~ N(0, I), o prior

    with torch.no_grad():
        samples = model.decode(z)

    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    show_digits(ax, tile(samples, rows=8, cols=8))

    ax.set_title(f"64 amostras do prior, $z \\sim \\mathcal{{N}}(0, I)$ — d = {LATENT_DIM}", pad=8)
    ax.set_xlabel("Cada célula é a decodificação de um z sorteado")

    fig.tight_layout()
    save(fig, "fig03-samples.png")
    # --8<-- [end:fig3]

    # -----------------------------------------------------------------
    # Numeros do relatorio
    # -----------------------------------------------------------------

    print("\nEXERCICIO 1 - VAE NO MNIST\n")

    print(f"treino: {len(train_set)} | validacao: {len(val_set)} | teste: {len(test_set)}")
    print(f"parametros do modelo: {sum(p.numel() for p in model.parameters())}\n")

    print("--- validacao por epoca ---")
    for epoch, (rec, kl_v) in enumerate(zip(history["val_recon"], history["val_kl"])):
        print(f"epoca {epoch:2d}: rec {rec:7.2f} | KL {kl_v:6.2f} | soma {rec + kl_v:7.2f}")

    epoch_peak = max(range(1, EPOCHS + 1), key=lambda e: history["val_kl"][e])
    print(f"\nKL de validacao: epoca 1 = {history['val_kl'][1]:.2f} | "
          f"epoca {EPOCHS} = {history['val_kl'][EPOCHS]:.2f} | "
          f"maximo na epoca {epoch_peak} = {history['val_kl'][epoch_peak]:.2f}")

    print("\n--- teste (nats por imagem) ---")
    print(f"reconstrucao: {test_recon:.2f}")
    print(f"KL          : {test_kl:.2f}")
    print(f"ELBO negativo: {test_recon + test_kl:.2f}")

    # Conferencia: a BCE escrita a mao bate com a do PyTorch (reduction="sum").
    with torch.no_grad():
        logits, _, _ = model(originals)
        mine = bce(logits, originals).sum().item()
        torch_bce = F.binary_cross_entropy(torch.sigmoid(logits), originals, reduction="sum").item()
    print(f"\nconferencia da BCE (10 imagens): mao = {mine:.4f} | torch = {torch_bce:.4f}")

    # Distancia media do prior aos mu do teste, para a analise do item E.
    with torch.no_grad():
        mu_test, logvar_test = model.encode(X_test)
    sigma_test = torch.exp(0.5 * logvar_test)
    print(f"\nnorma media de mu no teste: {mu_test.norm(dim=1).mean():.2f}")
    print(f"sigma medio por dimensao no teste: {sigma_test.mean(dim=0).numpy().round(3)}")
    print(f"desvio padrao de mu por dimensao: {mu_test.std(dim=0).numpy().round(3)}")
    print(f"indices das imagens da Figura 2: {first_idx}")

    # Onde caem as amostras do prior: distancia de cada z ao mu de treino
    # mais proximo, comparada com a de um mu de validacao (uma imagem real
    # que o modelo nao viu). So treino e validacao entram aqui.
    with torch.no_grad():
        mu_train, _ = model.encode(train_set.dataset.tensors[0][train_set.indices])
        mu_val, _ = model.encode(val_set.dataset.tensors[0][val_set.indices][:1000])
    torch.manual_seed(1)
    z_prior = torch.randn(1000, LATENT_DIM)

    # Dimensoes em que mu quase nao varia nao carregam informacao (o decoder
    # aprendeu a ignora-las); o prior so poe ruido nelas, entao ficam de fora.
    active = mu_train.std(dim=0) > 0.1
    mu_train, mu_val, z_prior = mu_train[:, active], mu_val[:, active], z_prior[:, active]

    d_val = torch.cdist(mu_val, mu_train).min(dim=1).values
    d_prior = torch.cdist(z_prior, mu_train).min(dim=1).values
    far = (d_prior > d_val.quantile(0.95)).float().mean()
    print(f"\ndimensoes ativas: {int(active.sum())} de {LATENT_DIM}")
    print(f"distancia ao mu de treino mais proximo (mediana): "
          f"validacao {d_val.median():.2f} | prior {d_prior.median():.2f}")
    print(f"amostras do prior mais longe que 95% das imagens de validacao: {far:.1%}")


if __name__ == "__main__":
    main()
