import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):

    if not seqs:
        return np.array([[]], dtype=int).reshape(0, 0)
    
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0

    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        trunc_seq = seq[:max_len]
        result[i, :len(trunc_seq)] = trunc_seq

    return result
    pass