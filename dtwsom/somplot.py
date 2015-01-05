import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sb


class InteractivePlot(object):

    def __init__(self, id, xlabel, ylabel):
        self.id = id
        plt.figure(id)
        plt.ion()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x = []
        self.y = []

    def update(self, xval, yval):
        x = self.x
        y = self.y
        x.append(xval)
        y.append(yval)
        plt.figure(self.id)
        plt.clf()
        plt.plot(x, y, 'k')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.draw()

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    matrix = matrix.T
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
    return fig

def plot_distance_map(matrix):
    fig = plt.figure()
    sb.heatmap(matrix)
    return fig

def grid_plot(matrix, width, height, normalize_scale=False, colors=None):
    fig = plt.figure(figsize=(8, 8))
    if colors is None:
        colors = ['blue'] * (width * height)
    outer_grid = gridspec.GridSpec(width, height, wspace=0.1, hspace=0.1)
    cnt = 0
    max_width = max(cell.shape[0] for row in matrix for cell in row if cell is not None)
    for i in range(width):
        for j in range(height):
            ax = plt.Subplot(fig, outer_grid[cnt])
            if matrix[i][j] is not None:
                sb.tsplot(matrix[i][j], color=colors[cnt], ax=ax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(-0.1, 1.1) 
                if normalize_scale:
                    ax.set_xlim(-1, max_width+1)
                fig.add_subplot(ax)
            cnt += 1
    return fig