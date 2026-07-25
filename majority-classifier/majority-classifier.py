import numpy as np

def majority_classifier(y_train, X_test):

    y_train = np.asarray(y_train)
    
    # Get unique classes, their first occurrence indices, and counts
    classes, first_indices, counts = np.unique(
        y_train, return_index=True, return_counts=True
    )
    
    # Sort classes and counts by order of first appearance in y_train
    appearance_order = np.argsort(first_indices)
    classes_in_order = classes[appearance_order]
    counts_in_order = counts[appearance_order]
    
    # np.argmax picks the first max count (preserving first-occurrence tie breaking)
    majority_class = classes_in_order[np.argmax(counts_in_order)]
    
    # Number of test samples
    n_test = len(X_test)
    
    # Return array filled with majority class
    return np.full(n_test, majority_class, dtype=int)
    pass