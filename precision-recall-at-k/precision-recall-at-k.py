def precision_recall_at_k(recommended, relevant, k):

    top_k = recommended[:k]
    relevant_set = set(relevant)
    
    # Count how many of the top-k recommended items are relevant
    hits = sum(1 for item in top_k if item in relevant_set)
    
    precision = hits / k
    recall = hits / len(relevant)
    
    return [float(precision), float(recall)]