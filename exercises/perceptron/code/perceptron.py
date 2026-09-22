"""Perceptron de camada unica, escrito do zero.


o Exercicio 1 (dados separaveis) e o Exercicio 2 (dados sobrepostos)
importam esta mesma classe, sem alterar nada. A unica diferenca entre os
dois usos e o argumento ``pocket`` de :meth:`Perceptron.fit`, que apenas
guarda uma copia dos melhores pesos ja vistos.

usando apenas numpy.
"""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------
# Ativacao
# ---------------------------------------------------------------------

# --8<-- [start:step]
def step(z: np.ndarray | float) -> np.ndarray:
    """Funcao degrau: 1 se z >= 0, 0 caso contrario.

    As classes sao rotuladas como {0, 1}, entao a saida da ativacao usa
    os mesmos valores dos rotulos.
    """

    return np.where(np.asarray(z) >= 0.0, 1, 0)
# --8<-- [end:step]


# ---------------------------------------------------------------------
# Resultado do treino
# ---------------------------------------------------------------------

# --8<-- [start:result]
@dataclass
class TrainingResult:
    """Tudo o que o relatorio precisa de uma rodada de treino."""

    # Pesos do ultimo iterado (o que o laco tiver ao final).
    w: np.ndarray
    b: float
    epochs: int
    accuracy: float
    converged: bool

    # Historico, uma entrada por epoca.
    accuracy_per_epoch: list[float] = field(default_factory=list)
    updates_per_epoch: list[int] = field(default_factory=list)

    # Preenchidos apenas quando fit(..., pocket=True).
    pocket_w: np.ndarray | None = None
    pocket_b: float | None = None
    pocket_accuracy: float | None = None
    pocket_epoch: int | None = None
    pocket_accuracy_per_epoch: list[float] = field(default_factory=list)
# --8<-- [end:result]


# ---------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------

# --8<-- [start:model]
class Perceptron:
    """Perceptron de camada unica com rotulos em {0, 1}.

    Predicao:

        yhat = step(w . x + b)

    Atualizacao, amostra a amostra:

        w <- w + eta * (y - yhat) * x
        b <- b + eta * (y - yhat)

    O erro (y - yhat) vale 0 quando a predicao esta certa (nenhuma
    atualizacao acontece), +1 quando um ponto da classe 1 foi previsto
    como 0 e -1 no erro oposto. E por isso que a regra funciona com
    rotulos {0, 1}: ela sabe corrigir os dois tipos de erro.
    """

    def __init__(
        self,
        w0: np.ndarray,
        b0: float = 0.0,
        learning_rate: float = 0.01,
    ) -> None:
        # copy() para que duas rodadas possam partir do MESMO w0 sem que
        # a primeira contamine a segunda.
        self.w = np.asarray(w0, dtype=float).copy()
        self.b = float(b0)
        self.learning_rate = float(learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Aplica step(X w + b) a todas as linhas de X de uma vez."""

        return step(np.asarray(X, dtype=float) @ self.w + self.b)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fracao de acertos no conjunto inteiro."""

        return float(np.mean(self.predict(X) == np.asarray(y)))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_epochs: int = 100,
        pocket: bool = False,
    ) -> TrainingResult:
        """Treina ate uma epoca inteira sem atualizacao, ou max_epochs.

        Com ``pocket=True``, sempre que uma atualizacao produz uma
        acuracia maior do que qualquer uma ja vista, os pesos daquele
        instante sao copiados para o "bolso" e guardados. E a unica
        coisa acrescentada ao laco -- o treino em si continua igual.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        accuracy_per_epoch: list[float] = []
        updates_per_epoch: list[int] = []
        pocket_per_epoch: list[float] = []

        # Bolso inicializado com os pesos iniciais (epoca 0).
        best_w = self.w.copy()
        best_b = self.b
        best_accuracy = self.accuracy(X, y)
        best_epoch = 0

        converged = False
        epochs_run = 0

        for epoch in range(1, max_epochs + 1):
            updates = 0

            for xi, yi in zip(X, y):
                # Predicao desta amostra, com os pesos DESTE instante.
                yhat = int(step(xi @ self.w + self.b))
                error = int(yi) - yhat

                # Amostra classificada corretamente: nada muda.
                if error == 0:
                    continue

                self.w = self.w + self.learning_rate * error * xi
                self.b = self.b + self.learning_rate * error
                updates += 1

                if pocket:
                    # Melhor-ate-agora: so a copia e acrescentada.
                    current = self.accuracy(X, y)

                    if current > best_accuracy:
                        best_accuracy = current
                        best_w = self.w.copy()
                        best_b = self.b
                        best_epoch = epoch

            epochs_run = epoch
            updates_per_epoch.append(updates)
            accuracy_per_epoch.append(self.accuracy(X, y))
            pocket_per_epoch.append(best_accuracy)

            # Criterio de parada: uma passada inteira sem atualizacao.
            if updates == 0:
                converged = True
                break

        return TrainingResult(
            w=self.w.copy(),
            b=self.b,
            epochs=epochs_run,
            accuracy=self.accuracy(X, y),
            converged=converged,
            accuracy_per_epoch=accuracy_per_epoch,
            updates_per_epoch=updates_per_epoch,
            pocket_w=best_w if pocket else None,
            pocket_b=best_b if pocket else None,
            pocket_accuracy=best_accuracy if pocket else None,
            pocket_epoch=best_epoch if pocket else None,
            pocket_accuracy_per_epoch=pocket_per_epoch if pocket else [],
        )
# --8<-- [end:model]
