import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads and cleans the Netflix Titles dataset.

    Reads the CSV file, fills missing descriptions with an empty string,
    filters out rows with missing titles, and drops duplicate entries.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        raise FileNotFoundError(f"Error reading {filepath}: {e}")

    # Drop rows with missing titles
    df = df.dropna(subset=['title'])

    # Fill missing descriptions
    if 'description' in df.columns:
        df['description'] = df['description'].fillna("")
    else:
        df['description'] = ""
    
    # Remove duplicate titles
    df = df.drop_duplicates(subset=['title'])

    # Optionally filter for movies only
    if 'type' in df.columns:
        df = df[df['type'].str.lower() == "movie"]

    logger.info("Data loaded and cleaned successfully.")
    return df
