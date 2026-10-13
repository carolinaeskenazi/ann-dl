"""MLP escrito do zero, usando apenas numpy.

Os Exercicios 2, 3 e 4 importam esta mesma classe, sem alterar nada. O
que muda de um exercicio para o outro sao apenas argumentos:

* ``layer_sizes`` -- o tamanho de cada camada, da entrada ate a saida;
* ``head`` -- o par (ativacao de saida, funcao de perda);
* ``learning_rate`` e o numero de epocas.

O forward, o backward e a atualizacao sao o mesmo codigo nos tres casos.

Convencao de formato: cada LINHA de X e uma amostra. Por isso os pesos da
camada l tem formato (n_entrada, n_saida) e a pre-ativacao e ``a @ W + b``
-- e a transposta da matriz W^(l) escrita no relatorio.
"""

import numpy as np


# ---------------------------------------------------------------------
# Saida + perda
# ---------------------------------------------------------------------

# --8<-- [start:heads]
class SigmoidBCE:
    """Saida sigmoide + entropia cruzada binaria (Exercicio 2).

    Y tem formato (N, 1), com valores em {0, 1}.
    """

    @staticmethod
    def activation(u: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-u))

    @staticmethod
    def loss(y_hat: np.ndarray, Y: np.ndarray) -> float:
        # clip evita log(0) quando a sigmoide satura.
        p = np.clip(y_hat, 1e-12, 1.0 - 1e-12)

        return float(-np.mean(Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p)))

    @staticmethod
    def delta(y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """dL/du da camada de saida: (y_hat - y) / N."""

        return (y_hat - Y) / len(Y)

    @staticmethod
    def labels(y_hat: np.ndarray) -> np.ndarray:
        return (y_hat >= 0.5).astype(int).ravel()


class SoftmaxCCE:
    """Saida softmax + entropia cruzada categorica (Exercicios 3 e 4).

    Y tem formato (N, K), em one-hot.
    """

    @staticmethod
    def activation(u: np.ndarray) -> np.ndarray:
        # Subtrair o maximo de cada linha nao muda o resultado e evita
        # overflow na exponencial.
        e = np.exp(u - u.max(axis=1, keepdims=True))

        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def loss(y_hat: np.ndarray, Y: np.ndarray) -> float:
        p = np.clip(y_hat, 1e-12, 1.0)

        return float(-np.mean(np.sum(Y * np.log(p), axis=1)))

    @staticmethod
    def delta(y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """dL/du da camada de saida: (y_hat - y) / N (derivado no relatorio)."""

        return (y_hat - Y) / len(Y)

    @staticmethod
    def labels(y_hat: np.ndarray) -> np.ndarray:
        return np.argmax(y_hat, axis=1)


class TanhMSE:
    """Saida tanh + erro quadratico medio.

    E a rede do Exercicio 1; usada apenas em ``exercise1_check.py`` para
    conferir a conta feita a mao.
    """

    @staticmethod
    def activation(u: np.ndarray) -> np.ndarray:
        return np.tanh(u)

    @staticmethod
    def loss(y_hat: np.ndarray, Y: np.ndarray) -> float:
        return float(np.mean(np.sum((Y - y_hat) ** 2, axis=1)))

    @staticmethod
    def delta(y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """dL/du = dL/dy_hat * (1 - y_hat^2), com dL/dy_hat = -2 (y - y_hat) / N."""

        return -2.0 * (Y - y_hat) / len(Y) * (1.0 - y_hat**2)
# --8<-- [end:heads]


# ---------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------

# --8<-- [start:model]
class MLP:
    """Perceptron multicamadas com camadas ocultas tanh.

    ``layer_sizes`` lista todas as camadas, da entrada ate a saida:
    ``[2, 8, 1]`` e uma rede com 2 entradas, 1 camada oculta de 8
    neuronios e 1 saida; ``[4, 16, 16, 3]`` tem 2 camadas ocultas.

    O treino e gradiente descendente em lote completo: a cada epoca, um
    forward e um backward sobre o conjunto de treino inteiro.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        head,
        rng: np.random.Generator,
        learning_rate: float = 0.1,
    ) -> None:
        self.layer_sizes = list(layer_sizes)
        self.head = head
        self.learning_rate = float(learning_rate)

        # Inicializacao de Xavier: desvio padrao 1/sqrt(n_entrada), para
        # que a tanh nao comece saturada. Os biases comecam em zero.
        self.W = [
            rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out))
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.b = [np.zeros(n_out) for n_out in layer_sizes[1:]]

    @property
    def n_parameters(self) -> int:
        return sum(W.size for W in self.W) + sum(b.size for b in self.b)

    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Devolve a ativacao de todas as camadas.

        ``activations[0]`` e a entrada X e ``activations[-1]`` e a saida
        y_hat. As intermediarias ficam guardadas porque o backward
        precisa delas.
        """

        activations = [np.asarray(X, dtype=float)]
        last = len(self.W) - 1

        for l, (W, b) in enumerate(zip(self.W, self.b)):
            u = activations[-1] @ W + b

            if l == last:
                activations.append(self.head.activation(u))
            else:
                activations.append(np.tanh(u))

        return activations

    def backward(
        self,
        activations: list[np.ndarray],
        Y: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagation: gradiente da perda em relacao a cada W e b."""

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        # delta = dL/du da camada atual, comecando pela saida.
        delta = self.head.delta(activations[-1], Y)

        for l in reversed(range(len(self.W))):
            # u = a @ W + b  =>  dL/dW = a^T delta,  dL/db = soma de delta.
            grads_W[l] = activations[l].T @ delta
            grads_b[l] = delta.sum(axis=0)

            if l > 0:
                # Volta uma camada: dL/da = delta W^T, e depois passa pela
                # derivada da tanh, 1 - tanh(u)^2 = 1 - a^2.
                delta = (delta @ self.W[l].T) * (1.0 - activations[l] ** 2)

        return grads_W, grads_b

    def update(
        self,
        grads_W: list[np.ndarray],
        grads_b: list[np.ndarray],
    ) -> None:
        """Um passo de gradiente descendente em todos os parametros."""

        for l in range(len(self.W)):
            self.W[l] -= self.learning_rate * grads_W[l]
            self.b[l] -= self.learning_rate * grads_b[l]

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int) -> list[float]:
        """Treina por ``epochs`` epocas e devolve a perda de treino de cada uma."""

        history = []

        for _ in range(epochs):
            activations = self.forward(X)
            history.append(self.head.loss(activations[-1], Y))

            grads_W, grads_b = self.backward(activations, Y)
            self.update(grads_W, grads_b)

        return history

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.head.loss(self.forward(X)[-1], Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.head.labels(self.forward(X)[-1])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
# --8<-- [end:model]
