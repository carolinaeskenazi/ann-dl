"""VAE totalmente conectado, a perda e o laco de treino.

Usado pelos Exercicios 1 (d = 16) e 2 (d = 2). Do PyTorch vem so o que o
enunciado permite: camadas ``nn.Linear``, ativacoes, Adam e autograd. A
reparametrizacao, os dois termos da perda e a amostragem estao escritos
aqui, sem nenhuma classe pronta de VAE nem ``kl_divergence``.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data import CACHE


# --8<-- [start:config]
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
BETA = 1.0
# --8<-- [end:config]


# --8<-- [start:model]
class VAE(nn.Module):
    """Encoder 784 -> 512 -> 256 -> (mu, log sigma^2); decoder d -> 256 -> 512 -> 784."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Duas cabecas lineares sobre a mesma representacao de 256.
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        # A ultima camada devolve logits; a sigmoide e aplicada em decode().
        # Separar as duas coisas deixa a BCE numericamente estavel (ver bce).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """z = mu + sigma * eps, com eps ~ N(0, I).

        O sorteio fica todo em eps, que nao depende de nenhum parametro; z e
        uma funcao deterministica e diferenciavel de mu e log sigma^2, entao
        o gradiente chega ao encoder.
        """

        sigma = torch.exp(0.5 * logvar)  # log sigma^2 -> sigma
        eps = torch.randn(mu.shape, generator=generator)

        return mu + sigma * eps

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Probabilidade de cada pixel, em (0, 1): e a imagem que se desenha."""

        return torch.sigmoid(self.decode_logits(z))

    def forward(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)

        return self.decode_logits(z), mu, logvar
# --8<-- [end:model]


# --8<-- [start:loss]
def bce(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """BCE somada sobre os 784 pixels, uma por imagem.

    Com p = sigmoid(l), o termo de cada pixel e
        -[x log p + (1 - x) log(1 - p)] = softplus(l) - x * l,
    que e a mesma conta sem nunca calcular log(0).
    """

    return (F.softplus(logits) - x * logits).sum(dim=1)


def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ) em forma fechada, somada sobre as d dimensoes.

    Por dimensao: 1/2 (sigma^2 + mu^2 - 1 - log sigma^2).
    """

    return 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(dim=1)


def vae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = BETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(perda, reconstrucao, KL), todos em nats por imagem.

    Soma sobre pixels / dimensoes e divide pelo tamanho do lote: o
    equivalente a reduction="sum" / B, e nao a reduction="mean", que
    tambem dividiria por 784 e mudaria o peso relativo do KL.
    """

    batch = x.shape[0]

    recon = bce(logits, x).sum() / batch
    kl_term = kl(mu, logvar).sum() / batch

    return recon + beta * kl_term, recon, kl_term
# --8<-- [end:loss]


# --8<-- [start:train]
@torch.no_grad()
def evaluate(model: VAE, dataset: Dataset) -> tuple[float, float]:
    """Reconstrucao e KL medios por imagem em um conjunto inteiro.

    z continua sendo sorteado (e o ELBO que se mede), mas de um gerador
    proprio com semente fixa: a avaliacao e reprodutivel e nao consome
    numeros do gerador global que o treino usa.
    """

    model.eval()
    generator = torch.Generator().manual_seed(0)

    total_recon, total_kl, n = 0.0, 0.0, 0
    for x, _ in DataLoader(dataset, batch_size=1000):
        logits, mu, logvar = model(x, generator)
        _, recon, kl_term = vae_loss(logits, x, mu, logvar)

        total_recon += recon.item() * len(x)
        total_kl += kl_term.item() * len(x)
        n += len(x)

    return total_recon / n, total_kl / n


def train(latent_dim: int, train_set: Dataset, val_set: Dataset) -> tuple[VAE, dict]:
    """Treina um VAE do zero e devolve o modelo e o historico por epoca."""

    torch.manual_seed(42)  # antes de construir o modelo, como pede o enunciado
    model = VAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    # A entrada "epoca 0" e a validacao do modelo antes de qualquer passo.
    val_recon, val_kl = evaluate(model, val_set)
    history = {
        "train_recon": [],
        "train_kl": [],
        "val_recon": [val_recon],
        "val_kl": [val_kl],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sum_recon, sum_kl, n = 0.0, 0.0, 0

        for x, _ in loader:
            logits, mu, logvar = model(x)
            loss, recon, kl_term = vae_loss(logits, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Media da epoca ponderada pelo tamanho de cada lote (o ultimo e menor).
            sum_recon += recon.item() * len(x)
            sum_kl += kl_term.item() * len(x)
            n += len(x)

        history["train_recon"].append(sum_recon / n)
        history["train_kl"].append(sum_kl / n)

        val_recon, val_kl = evaluate(model, val_set)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        print(
            f"d = {latent_dim:2d} | epoca {epoch:2d} | "
            f"treino: rec {history['train_recon'][-1]:7.2f}  KL {history['train_kl'][-1]:6.2f} | "
            f"validacao: rec {val_recon:7.2f}  KL {val_kl:6.2f}"
        )

    return model, history
# --8<-- [end:train]


# --8<-- [start:checkpoint]
def checkpoint_path(latent_dim: int):
    return CACHE / "vae" / f"vae_d{latent_dim}.pt"


def save_model(model: VAE, history: dict) -> None:
    path = checkpoint_path(model.latent_dim)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict(), "history": history}, path)


def load_or_train(latent_dim: int, train_set: Dataset, val_set: Dataset) -> tuple[VAE, dict]:
    """Reaproveita o modelo salvo por outro script; treina se ele nao existir."""

    path = checkpoint_path(latent_dim)
    if not path.exists():
        model, history = train(latent_dim, train_set, val_set)
        save_model(model, history)
        return model, history

    saved = torch.load(path)
    model = VAE(latent_dim)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    return model, saved["history"]
# --8<-- [end:checkpoint]
