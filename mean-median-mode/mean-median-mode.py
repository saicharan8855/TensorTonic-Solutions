import numpy as np
from collections import Counter

def mean_median_mode(x):

    x = np.asarray(x)

    mean = float(np.mean(x))
    median = float(np.median(x))

    counts = Counter(x)
    max_freq = max(counts.values())

    modes = [val for val , count in counts.items() if count == max_freq]
    mode = float(min(modes))

    return (mean, median, mode)
    pass