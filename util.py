import numpy as np

def neighboring(clusters):
    n_features, n_samples = clusters.shape
    offset = (0, -1, 1) 
    indices = ((i, j) for i in range(n_features) for j in range(n_samples))
    for i, j in indices:
        all_neigh = ((i + x, j + y) for x in offset for y in offset)
        valid = ((i*n_features + j) for i, j in all_neigh if (0 <= i < n_features) and (0 <= j < n_samples))
        target = valid.next()
        for neighbor in list(valid):
            yield targer, neighbor

