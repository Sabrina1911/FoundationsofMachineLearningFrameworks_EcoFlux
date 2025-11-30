from src.generate_data import generate_synthetic_energy_data


def test_dataset_shape():
    """Dataset must have >= 100 rows and exactly 4 columns."""
    df = generate_synthetic_energy_data()
    assert df.shape[1] == 4
    assert df.shape[0] >= 100


def test_column_names():
    """Dataset must contain expected columns."""
    df = generate_synthetic_energy_data()
    expected = {"layers", "training_hours", "flops_per_hour", "energy_kwh"}
    assert set(df.columns) == expected


def test_energy_is_positive():
    """Energy consumption must always be non-negative."""
    df = generate_synthetic_energy_data()
    assert (df["energy_kwh"] >= 0).all()
