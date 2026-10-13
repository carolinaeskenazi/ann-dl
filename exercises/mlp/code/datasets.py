"""Geracao e divisao dos dois conjuntos de dados da atividade.

Os parametros dos geradores sao os do enunciado, sem alteracao. O
scikit-learn e usado apenas aqui (gerar e dividir), na baseline linear e
na matriz de confusao.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

SEED = 42


# --8<-- [start:blobs]
def load_blobs():
    """Exercicio 2: classe 1 dividida em dois clusters, um de cada lado da classe 0."""

    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])

    X, c = make_blobs(
        n_samples=[500, 250, 250],
        centers=centers,
        cluster_std=1.2,
        random_state=SEED,
    )

    # classe 0: 1 cluster | classe 1: 2 clusters
    y = (c > 0).astype(int)

    return X, y
# --8<-- [end:blobs]


# --8<-- [start:multiclass]
def load_multiclass():
    """Exercicios 3 e 4: 1500 amostras, 4 features, 3 classes, 2 clusters por classe."""

    X, y = make_classification(
        n_samples=1500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=SEED,
    )

    return X, y
# --8<-- [end:multiclass]


# --8<-- [start:split]
def split(X: np.ndarray, y: np.ndarray):
    """80% treino / 20% teste, estratificado, com a mesma semente."""

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Rotulos inteiros -> matriz (N, K) com um 1 por linha."""

    return np.eye(n_classes)[y]
# --8<-- [end:split]
