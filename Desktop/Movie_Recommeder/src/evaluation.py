from typing import List

def precision_at_k(recommended: List[str], ground_truth: List[str], k: int) -> float:
    """Computes Precision@K.

    Args:
        recommended (List[str]): Recommended movie titles.
        ground_truth (List[str]): Ground truth relevant titles.
        k (int): Number of top recommendations.

    Returns:
        float: Precision@K value.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    recommended_at_k = recommended[:k]
    if not ground_truth:
        return 0.0

    relevant = len(set(recommended_at_k) & set(ground_truth))
    return relevant / k

def recall_at_k(recommended: List[str], ground_truth: List[str], k: int) -> float:
    """Computes Recall@K.

    Args:
        recommended (List[str]): Recommended movie titles.
        ground_truth (List[str]): Ground truth relevant titles.
        k (int): Number of top recommendations.

    Returns:
        float: Recall@K value.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    recommended_at_k = recommended[:k]
    if not ground_truth:
        return 0.0

    relevant = len(set(recommended_at_k) & set(ground_truth))
    return relevant / len(ground_truth)
