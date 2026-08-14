import numpy as np

def auc(fpr, tpr):

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    
    # Using np.trapezoid (or np.trapz in older NumPy versions)
    # trapezoid(y, x) computes integral of y along x using trapezoidal rule
    if hasattr(np, 'trapezoid'):
        area = np.trapezoid(tpr, fpr)
    else:
        area = np.trapz(tpr, fpr)
        
    return float(area)
    pass