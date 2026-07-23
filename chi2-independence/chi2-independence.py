import numpy as np

def chi2_independence(C):

    C = np.asarray(C, dtype=np.float64)
    
    # Calculate row, column, and total sums
    row_totals = np.sum(C, axis=1)      # Shape: (r,)
    col_totals = np.sum(C, axis=0)      # Shape: (c,)
    total = np.sum(C)                   # Scalar N
    
    # Compute expected frequencies using outer product: E_ij = (row_i * col_j) / total
    expected = np.outer(row_totals, col_totals) / total
    
    # Compute Chi-Square statistic: sum((O - E)^2 / E)
    chi2 = float(np.sum(((C - expected) ** 2) / expected))
    
    return chi2, expected
    
    pass