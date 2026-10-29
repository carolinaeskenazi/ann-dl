"""MNIST e a divisao treino / validacao / teste do enunciado.

As imagens sao carregadas uma vez como tensores (pixels em [0, 1], cada
imagem achatada em um vetor de 784), o que deixa o treino bem mais rapido do
que aplicar ``ToTensor`` imagem por imagem a cada epoca.

O download fica em ``.cache/mnist`` na raiz do repositorio, que o
``.gitignore`` ja exclui.
"""

from pathlib import Path

import torch
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets


# --8<-- [start:data]
ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / ".cache"

N_TRAIN, N_VAL = 50_000, 10_000


def to_tensors(train: bool) -> TensorDataset:
    """Um split oficial do MNIST como (X, y), com X de forma (N, 784) em [0, 1]."""

    mnist = datasets.MNIST(CACHE / "mnist", train=train, download=True)

    # data e uint8 (N, 28, 28): dividir por 255 da o mesmo que ToTensor.
    X = mnist.data.float().div(255.0).view(-1, 784)

    return TensorDataset(X, mnist.targets)


def load_mnist() -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Treino (50 000), validacao (10 000) e teste oficial (10 000)."""

    full_train = to_tensors(train=True)

    # O gerador com semente 42 fixa quais 10 000 imagens vao para a validacao.
    train_set, val_set = random_split(
        full_train,
        [N_TRAIN, N_VAL],
        generator=torch.Generator().manual_seed(42),
    )

    # O teste oficial so e usado para os numeros finais.
    test_set = to_tensors(train=False)

    return train_set, val_set, test_set
# --8<-- [end:data]
