import numpy as np

def matrix_trace(A):

    A = np.asarray(A)
    n = A.shape[0]

    trace_sum = 0

    for i in range(n):
        trace_sum += A[i, i]

    return trace_sum
    pass
