import numpy as np

def cosine_similarity(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    dot = np.dot(a,b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return dot/( a_norm * b_norm )
    pass