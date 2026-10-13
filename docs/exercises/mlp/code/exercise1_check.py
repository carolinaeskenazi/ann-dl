"""Exercise 1 - conferencia da conta feita a mao.

O Exercicio 1 esta resolvido passo a passo no relatorio. Este script
apenas repete o mesmo passo com a classe ``MLP`` e imprime os numeros,
para conferir as contas (e, de quebra, o backward usado nos Exercicios
2 a 4).

Uso, a partir da raiz do repositorio:

    python docs/exercises/mlp/code/exercise1_check.py
"""

import numpy as np

from mlp import MLP, TanhMSE

np.set_printoptions(precision=6, suppress=True)


def main() -> None:
    # --8<-- [start:check]
    x = np.array([[0.5, -0.2]])
    y = np.array([[1.0]])

    # O rng so e exigido pelo construtor: os pesos sao sobrescritos abaixo.
    model = MLP([2, 2, 1], head=TanhMSE, rng=np.random.default_rng(42), learning_rate=0.3)

    # A classe guarda W como (n_entrada, n_saida): a transposta do enunciado.
    model.W = [
        np.array([[0.3, -0.1], [0.2, 0.4]]).T,
        np.array([[0.5, -0.3]]).T,
    ]
    model.b = [np.array([0.1, -0.2]), np.array([0.2])]

    _, h1, y_hat = model.forward(x)
    loss_before = model.loss(x, y)

    grads_W, grads_b = model.backward([x, h1, y_hat], y)
    model.update(grads_W, grads_b)

    loss_after = model.loss(x, y)
    # --8<-- [end:check]

    print("EXERCICIO 1 - CONFERENCIA\n")

    print(f"h1     : {h1.ravel()}")
    print(f"y_hat  : {y_hat.item():.6f}")
    print(f"L      : {loss_before:.6f}\n")

    print(f"dL/dW2 : {grads_W[1].ravel()}")
    print(f"dL/db2 : {grads_b[1].item():.6f}")
    print(f"dL/dW1 :\n{grads_W[0].T}")
    print(f"dL/db1 : {grads_b[0]}\n")

    print(f"W2 novo: {model.W[1].ravel()}")
    print(f"b2 novo: {model.b[1].item():.6f}")
    print(f"W1 novo:\n{model.W[0].T}")
    print(f"b1 novo: {model.b[0]}\n")

    print(f"L depois da atualizacao: {loss_after:.6f}")


if __name__ == "__main__":
    main()
