import numpy as np

def single_node_gini(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities ** 2))
    
def gini_impurity(y_left, y_right):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    # Handle the case where both child nodes are empty
    if n_total == 0:
        return 0.0
    
    gini_left = single_node_gini(y_left)
    gini_right = single_node_gini(y_right)
    
    # Compute weighted average Gini impurity
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return float(weighted_gini)
    
    pass