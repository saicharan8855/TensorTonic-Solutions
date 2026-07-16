import numpy as np

def calculate_eigenvalues(matrix):

    try: 
        arr = np.asarray(matrix)
    except (ValueError, TypeError):
        return None

    if arr.size == 0 or arr.ndim != 2:
        return None

    if arr.shape[0] != arr.shape[1]:
        return None 

    eigenvalues = np.linalg.eigvals(arr)

    sort_indices = np.lexsort((eigenvalues.imag, eigenvalues.real))

    return eigenvalues[sort_indices]
    pass