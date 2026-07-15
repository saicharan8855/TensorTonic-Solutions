import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):


    pe = np.zeros((seq_len, d_model), dtype=float)
    position = np.arange(seq_len)[:, np.newaxis]
    
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    

    angles = position * div_term
    
    
    pe[:, 0::2] = np.sin(angles)
    
    if d_model > 1:
        pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
        
    return pe
    pass