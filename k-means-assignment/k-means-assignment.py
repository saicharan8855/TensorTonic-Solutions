def k_means_assignment(points, centroids):
    assignments = []
    
    for p in points:
        best_dist = float('inf')
        best_idx = 0
        
        for j, c in enumerate(centroids):
            squared_dist = sum((p_d - c_d) ** 2 for p_d, c_d in zip(p, c))
            
            if squared_dist < best_dist:
                best_dist = squared_dist
                best_idx = j
                
        assignments.append(best_idx)
        
    return assignments