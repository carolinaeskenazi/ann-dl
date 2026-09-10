"""Exercise 3 — Preparing Real-World Data for a Neural Network.

Prepara o train.csv do Spaceship Titanic para uma rede neural com tanh.

O script:
- descreve os dados;
- faz split 80/20 estratificado antes das transformacoes;
- imputa valores ausentes;
- cria TotalSpend;
- aplica log(1+x) aos gastos;
- faz one-hot encoding;
- normaliza variaveis numericas para [-1, 1];
- gera a Figura 6;
- verifica NaN, shape e intervalo final.

Nenhum modelo e treinado.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



BASE = Path(__file__).resolve().parents[1]

FIGURES = BASE / "figures"
DATA = r"C:\Users\carol\insper\6_semestre\train.csv"

rng = np.random.default_rng(42)


SPENDING = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

NUMERICAL = [
    "Age",
    *SPENDING,
]

CATEGORICAL = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
]

DROP_COLUMNS = [
    "Cabin",
    "Name",
    "PassengerId",
]


def make_encoder():
    """Encoder que ignora categorias novas no conjunto de teste."""

    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )


def main():
    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(DATA)

    # =============================================================
    # A — Get to know the data
    # =============================================================

    balance = df["Transported"].value_counts()

    positive_share = float(
        df["Transported"].mean()
    )

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": 100 * df.isna().mean(),
    })

    spending_stats = (
        df[SPENDING]
        .agg(["mean", "median", "max"])
        .T
    )

    print("\nCLASS BALANCE")
    print(balance)

    print(
        f"Proporção Transported=True: "
        f"{positive_share:.2%}"
    )

    print("\nMISSING VALUES")

    print(
        missing.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\nSPENDING STATISTICS")

    print(
        spending_stats.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # =============================================================
    # B — Split before you transform
    # =============================================================

    X = df.drop(
        columns=["Transported"]
    )

    y = (
        df["Transported"]
        .astype(int)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Valor pedido no Results summary:
    # ANTES de imputacao, log ou scaling.
    foodcourt_train_mean = float(
        X_train["FoodCourt"].mean()
    )

    foodcourt_train_median = float(
        X_train["FoodCourt"].median()
    )

    print("\nSPLIT")

    print(
        f"Treino: {X_train.shape[0]} amostras"
    )

    print(
        f"Teste: {X_test.shape[0]} amostras"
    )

    print(
        "FoodCourt no treino antes de transformar: "
        f"média={foodcourt_train_mean:.3f}, "
        f"mediana={foodcourt_train_median:.3f}"
    )

    # =============================================================
    # Remocao das colunas que o enunciado manda descartar.
    # =============================================================

    X_train = X_train.drop(
        columns=DROP_COLUMNS
    )

    X_test = X_test.drop(
        columns=DROP_COLUMNS
    )

    # =============================================================
    # C1 — Missing data
    #
    # Numericas: mediana.
    # Categoricas: moda.
    #
    # fit SOMENTE no treino.
    # =============================================================

    numerical_imputer = SimpleImputer(
        strategy="median"
    )

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    train_num = pd.DataFrame(
        numerical_imputer.fit_transform(
            X_train[NUMERICAL]
        ),
        columns=NUMERICAL,
    )

    test_num = pd.DataFrame(
        numerical_imputer.transform(
            X_test[NUMERICAL]
        ),
        columns=NUMERICAL,
    )

    train_cat = pd.DataFrame(
        categorical_imputer.fit_transform(
            X_train[CATEGORICAL]
        ),
        columns=CATEGORICAL,
    )

    test_cat = pd.DataFrame(
        categorical_imputer.transform(
            X_test[CATEGORICAL]
        ),
        columns=CATEGORICAL,
    )

    # =============================================================
    # C3 — Feature engineering: TotalSpend
    # =============================================================

    train_num["TotalSpend"] = (
        train_num[SPENDING].sum(axis=1)
    )

    test_num["TotalSpend"] = (
        test_num[SPENDING].sum(axis=1)
    )

    # =============================================================
    # C4 — Heavy tails: log(1+x)
    #
    # TotalSpend tambem recebe log para nao reintroduzir uma
    # variavel de gasto com cauda muito longa.
    # =============================================================

    log_columns = [
        *SPENDING,
        "TotalSpend",
    ]

    foodcourt_log_train = np.log1p(
        train_num["FoodCourt"].to_numpy()
    )

    train_num[log_columns] = np.log1p(
        train_num[log_columns]
    )

    test_num[log_columns] = np.log1p(
        test_num[log_columns]
    )

    # =============================================================
    # D1 — Figura 6
    # =============================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
    )

    for target, label in [
        (0, "Transported=False"),
        (1, "Transported=True"),
    ]:
        mask = (
            y_train.to_numpy()
            == target
        )

        raw_values = (
            X_train.loc[
                mask,
                "FoodCourt",
            ]
            .dropna()
            .to_numpy()
        )

        axes[0].hist(
            raw_values,
            bins=40,
            alpha=0.55,
            label=label,
        )

        axes[1].hist(
            foodcourt_log_train[mask],
            bins=40,
            alpha=0.55,
            label=label,
        )

    axes[0].set_title(
        "FoodCourt antes do pré-processamento"
    )
    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("Frequência")
    axes[0].legend()

    axes[1].set_title(
        "FoodCourt após $\\log(1+x)$"
    )
    axes[1].set_xlabel(
        "$\\log(1+FoodCourt)$"
    )
    axes[1].set_ylabel("Frequência")
    axes[1].legend()

    fig.suptitle(
        "Figura 6 — Efeito da transformação na cauda de FoodCourt"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "fig06-foodcourt-preprocessing.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =============================================================
    # C2 — Categorical features: One-Hot Encoding
    # =============================================================

    encoder = make_encoder()

    train_cat_encoded = encoder.fit_transform(
        train_cat
    )

    test_cat_encoded = encoder.transform(
        test_cat
    )

    # =============================================================
    # C5 — Scaling para [-1, 1]
    #
    # fit SOMENTE no treino.
    # clip=True impede valores do teste de escaparem da faixa.
    # =============================================================

    scaler = MinMaxScaler(
        feature_range=(-1, 1),
        clip=True,
    )

    train_num_scaled = scaler.fit_transform(
        train_num
    )

    test_num_scaled = scaler.transform(
        test_num
    )

    # Junta dados numericos e categoricos.
    X_train_final = np.hstack([
        train_num_scaled,
        train_cat_encoded,
    ])

    X_test_final = np.hstack([
        test_num_scaled,
        test_cat_encoded,
    ])

    # =============================================================
    # D2 — Final checks
    # =============================================================

    train_nan = int(
        np.isnan(
            X_train_final
        ).sum()
    )

    test_nan = int(
        np.isnan(
            X_test_final
        ).sum()
    )

    train_min = float(
        X_train_final.min()
    )

    train_max = float(
        X_train_final.max()
    )

    test_min = float(
        X_test_final.min()
    )

    test_max = float(
        X_test_final.max()
    )

    print("\nFINAL CHECKS")

    print(
        f"NaN no treino: {train_nan}"
    )

    print(
        f"NaN no teste: {test_nan}"
    )

    print(
        f"Shape final treino: "
        f"{X_train_final.shape}"
    )

    print(
        f"Shape final teste: "
        f"{X_test_final.shape}"
    )

    print(
        f"Faixa treino: "
        f"[{train_min:.3f}, {train_max:.3f}]"
    )

    print(
        f"Faixa teste: "
        f"[{test_min:.3f}, {test_max:.3f}]"
    )

    print(
        f"Número de features finais: "
        f"{X_train_final.shape[1]}"
    )


if __name__ == "__main__":
    main()