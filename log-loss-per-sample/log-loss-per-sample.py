import math

def log_loss(y_true, y_pred, eps=1e-15):
    losses = []
    for y, p in zip(y_true, y_pred):
        # Clip p to the range [eps, 1 - eps]
        p_hat = min(max(p, eps), 1 - eps)
        
        # Binary cross-entropy formula
        loss = -(y * math.log(p_hat) + (1 - y) * math.log(1 - p_hat))
        losses.append(loss)
        
    return losses
    pass