import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):

    x = np.asarray(x)
    n = len(x)
    
    # Generate random sample indices
    if rng is not None:
        indices = rng.integers(0, n, size=(n_bootstrap, n))
    else:
        indices = np.random.randint(0, n, size=(n_bootstrap, n))
    
    # Resample and calculate mean for each bootstrap sample
    boot_samples = x[indices]
    boot_means = np.mean(boot_samples, axis=1)
    
    # Compute confidence interval using percentiles
    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1.0 - alpha)
    
    return boot_means, lower, upper
    pass
