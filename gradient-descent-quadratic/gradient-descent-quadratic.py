def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    X = x0
    for _ in range(steps):
        grad = 2*a*X + b
        X = X-lr*grad 
    return float(X)
        
    
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    pass