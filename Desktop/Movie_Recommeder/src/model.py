import logging
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src import config

logger = logging.getLogger(__name__)

class RecommendationModel:
    """Content-based movie recommendation model using TF-IDF vectorization and cosine similarity.

    Attributes:
        data (pd.DataFrame): Movie dataset.
        tfidf_matrix: TF-IDF feature matrix.
        cosine_sim (ndarray): Cosine similarity matrix.
        indices (pd.Series): Mapping of movie titles to DataFrame indices.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): DataFrame with 'title' and 'description' columns.
        """
        self.data = data.reset_index(drop=True)
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()

    def build_model(self) -> None:
        """Builds the TF-IDF model and computes the cosine similarity matrix."""
        logger.info("Initializing TF-IDF vectorizer with parameters: %s", config.TFIDF_PARAMS)
        tfidf = TfidfVectorizer(**config.TFIDF_PARAMS)
        
        descriptions = self.data['description'].fillna("")
        self.tfidf_matrix = tfidf.fit_transform(descriptions)
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        logger.info("Model built successfully. Cosine similarity matrix computed.")

    def get_recommendations(self, title: str, top_n: int = 10) -> List[str]:
        """Generates a list of movie recommendations similar to the given title.

        Args:
            title (str): Base movie title for recommendations.
            top_n (int, optional): Number of recommendations. Defaults to 10.

        Returns:
            List[str]: List of recommended movie titles.
        """
        if self.cosine_sim is None:
            logger.error("Model not built. Call build_model() first.")
            raise ValueError("Model has not been built. Please call build_model() first.")

        if title not in self.indices:
            logger.warning("Title '%s' not found in dataset.", title)
            return []

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: top_n + 1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = self.data['title'].iloc[movie_indices].tolist()
        logger.info("Generated %d recommendations for title '%s'.", len(recommendations), title)
        return recommendations
