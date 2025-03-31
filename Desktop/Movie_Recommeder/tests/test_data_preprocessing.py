import pandas as pd
import pytest
from src.data_preprocessing import load_and_clean_data

SAMPLE_CSV = """title,description,type
Movie A,"A great movie",Movie
Movie B,,"Movie"
Movie C,"Another movie",Movie
"""

def test_load_and_clean_data(tmp_path):
    file = tmp_path / "sample.csv"
    file.write_text(SAMPLE_CSV)
    
    df = load_and_clean_data(str(file))
    
    # Ensure missing descriptions are filled with an empty string
    assert df.loc[df['title'] == "Movie B", "description"].iloc[0] == ""
    # No missing titles should be present
    assert df['title'].isnull().sum() == 0
    # All rows should be movies
    assert all(df['type'].str.lower() == "movie")
