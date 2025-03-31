import pandas as pd
import pytest
from src.model import RecommendationModel

@pytest.fixture
def sample_data():
    data = {
        "title": ["Movie A", "Movie B", "Movie C"],
        "description": ["A great movie", "An awesome film", "A fantastic tale"]
    }
    return pd.DataFrame(data)

def test_build_model(sample_data):
    model = RecommendationModel(sample_data)
    model.build_model()
    
    assert model.tfidf_matrix is not None
    assert model.cosine_sim is not None
    n = sample_data.shape[0]
    assert model.cosine_sim.shape == (n, n)

def test_get_recommendations(sample_data):
    model = RecommendationModel(sample_data)
    model.build_model()
    
    recs = model.get_recommendations("Movie A", top_n=2)
    assert "Movie A" not in recs
    assert len(recs) <= 2

def test_title_not_found(sample_data):
    model = RecommendationModel(sample_data)
    model.build_model()
    
    recs = model.get_recommendations("Nonexistent")
    assert recs == []
