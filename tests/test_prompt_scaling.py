from app import prompt_scaling_factor


def test_prompt_scaling_short():
    """Token count <= 8 → scaling = 1.00"""
    assert prompt_scaling_factor(5) == 1.00


def test_prompt_scaling_medium():
    """Token count 9–20 → scaling = 1.05"""
    assert prompt_scaling_factor(10) == 1.05


def test_prompt_scaling_long():
    """Token count > 20 → scaling = 1.10"""
    assert prompt_scaling_factor(50) == 1.10
