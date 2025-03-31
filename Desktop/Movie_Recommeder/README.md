# Movie Recommendation System

## Overview

This project implements a movie recommendation system using a content-based filtering approach. 
It leverages the [Netflix Titles dataset](https://www.kaggle.com/shivamb/netflix-shows) to build a model that recommends similar movies based on their descriptions using TF-IDF vectorization and cosine similarity.

## Features

- **Data Loading & Cleaning:** Robust routines to load and preprocess the Netflix Titles dataset.
- **Model Building:** A `RecommendationModel` class that builds the TF-IDF model and computes cosine similarity between movie descriptions.
- **Evaluation:** Utility functions to compute evaluation metrics (Precision@K and Recall@K).
- **Configurable Parameters:** Easily adjustable parameters via a configuration module.
- **Demo CLI:** A demo script (`demo.py`) that allows users to input a movie title via command-line arguments.
- **Testing:** Unit tests for core functionalities using `pytest`.
- **Containerization:** A Dockerfile to build and run the application in a container.
- **CI/CD:** A sample GitHub Actions workflow for automated testing.

## Project Structure
