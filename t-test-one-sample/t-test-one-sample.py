import numpy as np

def t_test_one_sample(x, mu0):
    x = np.asarray(x, dtype=np.float64)
    
    # Calculate sample size
    n = len(x)
    
    # Compute sample mean
    x_bar = np.mean(x)
    
    # Compute sample standard deviation with Bessel's correction (ddof=1 for n - 1)
    s = np.std(x, ddof=1)
    
    # Compute standard error of the mean
    se = s / np.sqrt(n)
    
    # Calculate t-statistic
    t_stat = (x_bar - mu0) / se
    
    return float(t_stat)
    pass