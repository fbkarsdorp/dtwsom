import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    fig, ax = plt.subplots()
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    ax.invert_xaxis()

def plot_distance_map(matrix, interpolation='bilinear', ax=None):
    fig = plt.figure()
    plt.imshow(matrix, interpolation=interpolation, cmap=cm.RdYlGn,
              vmax=abs(matrix).max(), vmin=-abs(matrix).max())