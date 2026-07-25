import numpy as np

def entropy_node(y):


    if len(y) == 0:
        return 0.0
    
    # Calculate class counts
    _, counts = np.unique(y, return_counts=True)
    
    # Calculate probabilities
    probs = counts / len(y)
    
    # Filter out zero probabilities to avoid log2(0)
    probs = probs[probs > 0]
    
    # Compute Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)
    pass