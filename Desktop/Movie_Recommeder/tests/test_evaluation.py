import pytest
from src.evaluation import precision_at_k, recall_at_k

def test_precision_at_k_normal():
    recommended = ["Movie A", "Movie B", "Movie C"]
    ground_truth = ["Movie A", "Movie D"]
    prec = precision_at_k(recommended, ground_truth, 3)
    assert prec == pytest.approx(1/3)

def test_recall_at_k_normal():
    recommended = ["Movie A", "Movie B", "Movie C"]
    ground_truth = ["Movie A", "Movie D"]
    rec = recall_at_k(recommended, ground_truth, 3)
    assert rec == pytest.approx(0.5)

def test_precision_empty_ground_truth():
    prec = precision_at_k(["Movie A", "Movie B"], [], 2)
    assert prec == 0.0

def test_recall_empty_ground_truth():
    rec = recall_at_k(["Movie A", "Movie B"], [], 2)
    assert rec == 0.0

def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["Movie A"], ["Movie A"], 0)
