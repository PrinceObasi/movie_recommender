"""
Configuration file for the movie recommendation system.
Adjust the parameters as needed.
"""

# TF-IDF Vectorizer parameters
TFIDF_PARAMS = {
    "stop_words": "english",
    "max_df": 0.85,
    "min_df": 2
}

# Model settings
MODEL_SETTINGS = {
    "default_top_n": 10
}
