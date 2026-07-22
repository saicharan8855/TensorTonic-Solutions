import numpy as np

def poisson_pmf_cdf(lam, k):

    i = np.arange(k + 1)
    
    # Compute log factorial values stably: log(0!) = 0, log(i!) = sum(log(1..i))
    log_fact = np.zeros(k + 1)
    if k > 0:
        log_fact[1:] = np.cumsum(np.log(np.arange(1, k + 1)))
    
    # log(PMF(i)) = -lam + i * log(lam) - log(i!)
    log_pmfs = -lam + i * np.log(lam) - log_fact
    pmfs = np.exp(log_pmfs)
    
    pmf = float(pmfs[-1])
    cdf = float(np.sum(pmfs))
    
    return pmf, cdf
    pass