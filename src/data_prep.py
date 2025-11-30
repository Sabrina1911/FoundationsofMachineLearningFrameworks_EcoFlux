import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("../data/energy_synthetic.csv")

def load_energy_dataset(test_size=0.2, random_state=1911):
    """
    Load the synthetic energy dataset from CSV and split into train/test.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X = df[["num_layers", "training_hours", "flops_per_hour"]]
    y = df["energy_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
