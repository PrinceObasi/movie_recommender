"""
Demo script to run the movie recommendation system.
Allows specifying a movie title via command-line arguments.
"""

import argparse
import logging

from src.data_preprocessing import load_and_clean_data
from src.model import RecommendationModel

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System Demo")
    parser.add_argument('--title', type=str, default="The Irishman", help="Movie title to base recommendations on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Path to the Netflix Titles CSV file
    data_path = "netflix_titles.csv"
    
    logger.info("Loading and cleaning data...")
    data = load_and_clean_data(data_path)
    
    logger.info("Building recommendation model...")
    recommender = RecommendationModel(data)
    recommender.build_model()
    
    sample_title = args.title
    logger.info(f"Generating recommendations for: {sample_title}")
    recommendations = recommender.get_recommendations(sample_title, top_n=10)
    
    if recommendations:
        for idx, title in enumerate(recommendations, start=1):
            print(f"{idx}. {title}")
    else:
        print("No recommendations found. Please check the input title.")

if __name__ == "__main__":
    main()
