def cosine_embedding_loss(x1, x2, label, margin):
    dot_product = sum(a * b for a, b in zip(x1, x2))
    
    norm_x1 = math.sqrt(sum(a ** 2 for a in x1))
    norm_x2 = math.sqrt(sum(b ** 2 for b in x2))
    
    cos_sim = dot_product / (norm_x1 * norm_x2)
    
    if label == 1:
        return 1.0 - cos_sim
    else:
        return max(0.0, cos_sim - margin)
    