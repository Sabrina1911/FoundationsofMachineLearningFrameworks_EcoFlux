import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_energy_data(n_samples: int = 120, random_state: int = 1911) -> pd.DataFrame:
    """
    Generate a synthetic dataset for ML energy estimation.

    Columns:
    - layers
    - training_hours
    - flops_per_hour
    - energy_kwh
    """
    rng = np.random.default_rng(random_state)

    layers = rng.integers(2, 25, size=n_samples)
    training_hours = rng.uniform(1.0, 24.0, size=n_samples)
    flops_per_hour = rng.uniform(20.0, 300.0, size=n_samples)

    # Simple linear-ish relationship with some noise
    base_energy = (
        0.08 * layers +
        0.12 * training_hours +
        0.015 * flops_per_hour
    )

    noise = rng.normal(loc=0.0, scale=0.5, size=n_samples)
    energy_kwh = np.clip(base_energy + noise, a_min=0.0, a_max=None)

    df = pd.DataFrame(
        {
            "layers": layers,
            "training_hours": training_hours,
            "flops_per_hour": flops_per_hour,
            "energy_kwh": energy_kwh,
        }
    )

    return df


def save_synthetic_dataset(
    output_path: Path = Path("../data/energy_synthetic.csv"),
    n_samples: int = 120,
    random_state: int = 1911,
) -> Path:
    """
    Generate and save the synthetic dataset to CSV.
    """
    df = generate_synthetic_energy_data(n_samples=n_samples, random_state=random_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = save_synthetic_dataset()
    print(f"[✔] Generated dataset saved to: {path}")
