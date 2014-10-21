import numpy as np

def normalize(signal, minimum=None, maximum=None):
    """Normalize a signal to the range 0, 1"""
    signal = np.array(signal).astype('float')
    if minimum is None:
        signal -= np.min(signal)
    else:
        signal -= minimum
    if maximum is None:
        signal /= np.max(signal)
    else:
        signal /= maximum - minimum
    signal = np.clip(signal, 0.0, 1.0)
    return signal

def neighboring(clusters):
    n_features, n_samples = clusters.shape
    offset = (0, -1, 1) 
    indices = ((i, j) for i in range(n_features) for j in range(n_samples))
    for i, j in indices:
        all_neigh = ((i + x, j + y) for x in offset for y in offset)
        valid = ((i*n_features + j) for i, j in all_neigh if (0 <= i < n_features) and (0 <= j < n_samples))
        target = valid.next()
        for neighbor in list(valid):
            yield target, neighbor

