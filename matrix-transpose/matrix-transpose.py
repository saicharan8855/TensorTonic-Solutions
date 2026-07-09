import numpy as np

def matrix_transpose(A):
    row_len = len(A)
    col_len = len(A[0])

    result = []

    for col in range(col_len):
        new_row = []
        for row in range(row_len):
            new_row.append(A[row][col])
        result.append(new_row)
        
    result = np.asarray(result, dtype = float)
    return result
    pass
