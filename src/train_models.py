import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_prep import load_energy_dataset

MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "R2": r2_score(y_test, preds)
    }

    return model, metrics


def train_mlp_regressor(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16),
                       max_iter=600,
                       random_state=1911)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp.fit(X_train_scaled, y_train)
    preds = mlp.predict(X_test_scaled)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "R2": r2_score(y_test, preds)
    }

    return mlp, scaler, metrics



if __name__ == "__main__":
    print("[1/3] Loading dataset…")
    X_train, X_test, y_train, y_test = load_energy_dataset()

    print("[2/3] Training Linear Regression…")
    lin_model, lin_metrics = train_linear_regression(X_train, y_train, X_test, y_test)

    print("[✔] Linear Regression metrics:", lin_metrics)

    print("[2/3] Training MLPRegressor…")
    mlp_model, mlp_scaler, mlp_metrics = train_mlp_regressor(X_train, y_train, X_test, y_test)

    print("[✔] MLPRegressor metrics:", mlp_metrics)

    print("[3/3] Saving models…")

    joblib.dump(lin_model, MODELS_DIR / "ecoflux_linear_regression.pkl")
    joblib.dump(
        {"model": mlp_model, "scaler": mlp_scaler},
        MODELS_DIR / "ecoflux_mlp_regressor.pkl"
    )

    print("[✔] Models saved to /models/")
