from __future__ import division

import cPickle
import logging
import math
import random

from collections import defaultdict
from functools import partial
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sb

from scipy.interpolate import interp1d
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph

from dtw import dtw_distance, average_sequence
from dgw.dtw.scaling import uniform_scaling_to_length, uniform_shrinking_to_length
from somplot import plot_distance_map, hinton, grid_plot, InteractivePlot


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def median_example(X, dist_fn):
    if len(X) == 1:
        return 0
    dm = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i):
            dm[j, i] = dm[i, j] = dist_fn(X[i], X[j])
    return np.argmin(dm.sum(1) / dm.shape[0])

def no_decay(start):
    while True:
        yield start

def linear_decay(start, end, iterations):
    return iter(np.linspace(start, end, iterations))

def exponential_decay(start, end, iterations, k=2):
    return iter(start * np.power(k, (np.arange(iterations) / float(iterations)) * np.log(end / start) * 1 / np.log(k)))

class Som(object):
    def __init__(self, x, y, input_len=0, sigma=None, sigma_end=None, eta=0.5, eta_end=0.01, 
                 decay_fn="exponential", random_seed=None, n_iter=1000, intermediate_plots=False, 
                 compute_errors=250):

        """Initializes a Self Organizing Maps.
        x, y - dimensions of the SOM
        input_len - number of the elements of the vectors in input
        sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
        learning_rate - initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
        random_seed, random seed to use."""

        if sigma >= x / 2.0 or sigma >= y / 2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        self.random_seed = random_seed
        self.random_generator = np.random.RandomState(random_seed)
        self.activation_map = np.zeros((x, y))
        self.x, self.y = x, y
        self.n_iter = n_iter
        self.sigma = sigma if sigma is not None else np.ceil(1 + np.floor(min(self.x, self.y)-1)/2.)-1

        if decay_fn == 'exponential':
            self.sigma_decay = exponential_decay(self.sigma, sigma_end, n_iter)
            self.eta_decay = exponential_decay(eta, eta_end, n_iter)
        elif decay_fn == 'linear':
            self.sigma_decay = linear_decay(sigma, sigma_end, n_iter)
            self.eta_decay = linear_decay(eta, eta_end, n_iter)
        elif decay_fn == 'constant':
            self.sigma_decay = no_decay(sigma)
            self.eta_decay = no_decay(eta)

        self.xx, self.yy = np.meshgrid(np.arange(x), np.arange(y))
        self.neigx = np.arange(x)
        self.neigy = np.arange(y)
        self.compute_errors = compute_errors
        self.intermediate_plots = intermediate_plots

        self._init_weights(x, y, input_len)

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix the element i,j 
        is the response of the neuron i,j to x."""
        s = np.subtract(x, self.weights) # x - w
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = np.linalg.norm(s[it.multi_index]) # || x - w ||
            it.iternext()

    def activate(self,x):
        "Returns the activation map to x."
        self._activate(x)
        return self.activation_map

    def gaussian(self,c,sigma):
        """ Returns a Gaussian centered in c """
        d = 2.0 * sigma * sigma #* np.pi
        ax = np.exp(-np.power(self.neigx - c[0], 2) / d)
        ay = np.exp(-np.power(self.neigy - c[1], 2) / d)
        return np.outer(ax, ay) # the external product gives a matrix

    def winner(self, x):
        "Computes the coordinates of the winning neuron for the sample x."
        self._activate(x)
        return np.unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    def update(self, x, win, t):
        """Updates the weights of the neurons.
        x - current pattern to learn;
        win - position of the winning neuron for x (array or tuple);
        t - iteration index."""
        eta = self.eta_decay.next()
        sig = self.sigma_decay.next()
        logging.info('Eta = %.4f / Sigma = %.4f' % (eta, sig))
        g = self.gaussian(win, sig) * eta # improves the performances
        it = np.nditer(g, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] += g[it.multi_index] * (x - self.weights[it.multi_index])
            self.weights[it.multi_index] = self.weights[it.multi_index] / np.linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron) 
        to each sample in data."""
        q = np.zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q

    def _init_weights(self, x, y, input_len):
        self.weights = self.random_generator.rand(x, y, input_len) * 2 - 1 # random initialization
        self.weights = np.array([v / np.linalg.norm(v) for v in self.weights]) # normalization

    def random_weights_init(self, data):
        "Initializes the weights of the SOM picking random samples from data."
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[self.random_generator.randint(len(data))].copy()
            self.weights[it.multi_index] = self.weights[it.multi_index] / np.linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self, data):
        "Trains the SOM picking samples at random from data."
        if self.compute_errors:
            quantization_fig = InteractivePlot(0, "iterations", "quantization error")
            topology_fig = InteractivePlot(1, "iterations", "topology error")            
        for iteration in range(self.n_iter):
            logging.info('Iteration %d / %d' % (iteration + 1, self.n_iter))
            rand_i = self.random_generator.randint(len(data))
            # logging.debug("Input selection: %d" % rand_i)
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)
            if self.compute_errors and iteration % self.compute_errors == 0:
                quantization_fig.update(iteration, self.quantization_error(data))
                topology_fig.update(iteration, self.topology_error(data))

    def train_batch(self, data, shuffle=False):
        "Trains using all the vectors in data sequentially."
        self.random_generator.shuffle(data)
        if self.compute_errors:
            quantization_fig = InteractivePlot(0, "iterations", "quantization error")
            topology_fig = InteractivePlot(1, "iterations", "topology error") 
        iteration = 0
        while iteration < self.n_iter:
            logging.info('Iteration %d / %d' % (iteration + 1, self.n_iter))
            idx = iteration % (len(data) - 1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            if self.compute_errors and iteration % self.compute_errors == 0:
                quantization_fig.update(iteration, self.quantization_error(data))
                topology_fig.update(iteration, self.topology_error(data))
            iteration += 1

    def distance_map(self):
        """Returns the average distance map of the weights.
        (Each mean is normalized in order to sum up to 1)."""
        um = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += np.linalg.norm(self.weights[ii, jj, :] - self.weights[it.multi_index])
            it.iternext()
        um = um / um.max()
        return um

    def distance_matrix(self, dist_fn=lambda a, b: np.linalg.norm(a - b)):
        neurons = [self.weights[i][j] for i in range(self.x) for j in range(self.y)]
        n = self.x * self.y
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                dm[j, i] = dm[i, j] = dist_fn(neurons[i], neurons[j])
        return dm

    def activation_response(self, data):
        """Returns a matrix where the element i,j is the number of times
        that the neuron i,j have been winner."""
        a = np.zeros((self.x, self.y))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def topology_error(self, data):
        error = 0
        for x in data:
            activation_map = self.activate(x)
            indices = activation_map.ravel().argsort()[:2]
            indices = [np.unravel_index(i, activation_map.shape) for i in indices]
            u, v = indices[0], indices[1]
            if math.sqrt(sum((a - b)**2 for a, b in zip(u, v))) > 1:
                error += 1.0
        return error / len(data)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average distance between
        each input sample and its best matching unit."""
        error = 0
        for x in data:
            i, j = self.winner(x)
            error += np.linalg.norm(x - self.weights[i][j])
        return error / len(data)

    def win_map(self, data, dist=False, labels=None):
        """Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
        that have been mapped in the position i,j."""
        winmap = defaultdict(list)
        for k, x in enumerate(data):
            assert x.max() > 0
            i, j = self.winner(x)
            if dist:
                winmap[i, j].append((x, self.activation_map[i, j]))
            elif labels is not None:
                winmap[i, j].append(labels[k])
            else:
                winmap[i, j].append(x)
        return dict(winmap)

    def _cluster(self, linkage='complete', k=6):
        C = grid_to_graph(self.x, self.y)
        X = np.array(self.weights).reshape((self.x * self.y, self.weights[0][0].shape[0]))
        clusterer = AgglomerativeClustering(n_clusters=k, connectivity=C, affinity=self.dtw_fn, linkage=linkage)
        return clusterer.fit_predict(X)

    def _plot_grid(self, what='prototypes', data=None, normalize_scale=True, colors=None):
        if what == 'prototypes':
            points = self.weights
        elif what == 'obs':
            if data is None:
                raise ValueError("Must supply raw data points.")
            points = [[None for i in range(self.x)] for j in range(self.y)]
            wm = self.win_map(data)
            for (i, j), value in wm.iteritems():
                points[i][j] = value[median_example(value, self.dtw_fn)]
        return grid_plot(points, self.x, self.y, normalize_scale, colors=colors)

    def plot(self, t=0, what='prototypes', data=None, kind='grid', clusters=0, normalize_scale=True, fp="", close=False):
        if kind == 'grid':
            if clusters:
                colors = sb.color_palette("deep", clusters)
                colors = [colors[c] for c in self._cluster(k=clusters)]
            else:
                colors = None
            fig = self._plot_grid(what=what, data=data, normalize_scale=normalize_scale, colors=colors)
        elif kind == 'dmap':
            fig = plot_distance_map(self.distance_map())
        elif kind == 'hinton':
            if data is None:
                raise ValueError("Must supply raw data points.")
            ar = self.activation_response(data)
            fig = hinton(ar / np.linalg.norm(ar))
        else:
            raise ValueError("Plot type '%s' is not supported." % kind)
        if fp:
            plt.savefig(fp)
        if close:
            plt.close(fig)

    def save(self, filepath):
        "Serialize a model to disk."
        with open(filepath, "wb") as out:
            cPickle.dump(self, out, protocol=cPickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath):
        "Load a saved model."
        with open(filepath, "rb") as infile:
            model = cPickle.load(infile)
        return model


class DtwSom(Som):
    def __init__(self, x, y, sigma=None, sigma_end=0.5, eta=0.5, eta_end=0.01,  
                 decay_fn="exponential", random_seed=None, n_iter=1000, intermediate_plots=False, 
                 compute_errors=500, constraint='slanted_band', metric='seuclidean', window=5,
                 step_pattern=2, normalized=True, update_fn="som"):

        super(DtwSom, self).__init__(x, y, sigma=sigma, sigma_end=sigma_end, eta=eta, eta_end=eta_end,
                                     decay_fn=decay_fn, random_seed=random_seed, n_iter=n_iter,
                                     intermediate_plots=intermediate_plots, compute_errors=compute_errors)

        self.dtw_fn = partial(dtw_distance, constraint=constraint,
                                            window=window,
                                            metric=metric,
                                            step_pattern=step_pattern,
                                            normalized=normalized)
        self._update_weights = self._som_update if update_fn == 'som' else self._average_sequence_update

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix the element i,j
        is the response of the neuron i,j to x """
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            i, j = it.multi_index
            if self.weights[i][j].size == 0:
                # make a new neuron with random signals of a random size
                neuron = np.array(self.random_generator.randint(2, size=x.shape[0]), dtype=np.float)
                self.weights[i][j] = neuron / np.linalg.norm(neuron)
            # compute the distance between this neuron and the input 
            self.activation_map[it.multi_index] = self.dtw_fn(self.weights[i][j], x)
            it.iternext()

    def update(self, ts, win, t):
        eta = self.eta_decay.next()
        sig = self.sigma_decay.next()
        logging.info('Eta = %.4f / Sigma = %.4f' % (eta, sig))

        g = self.gaussian(win, sig) * eta        
        for i in range(self.x):
            for j in range(self.y):
                h = g[i, j]
                if h > 0:
                    self.weights[i][j] = self._update_weights(self.weights[i][j], ts, (1-h), h)

    def _som_update(self, M, X, hm, hx):
        _, _, (x_arr, y_arr) = self.dtw_fn(X, M, dist_only=False)
        M_ = M.copy() # we must update t+1, so first make a copy...
        for p, (i, j) in enumerate(zip(x_arr, y_arr)):
            M_[j] += hx * (X[i] - M[j])
        return M_ / np.linalg.norm(M_)

    def _average_sequence_update(self, M, X, hm, hx):
        """Compute an average sequence between two input sequences based on the warping path
        between them and using linear interpolation for compression."""
        # In the model adaptation the length of the new model vector sequence is determined first
        avg_length = int(0.5 + round(hm * M.shape[0] + hx * X.shape[0]))
        # control for averages too far away... (Ugly hack...)
        if not ((M.shape[0] <= avg_length <= X.shape[0]) or (X.shape[0] <= avg_length <= M.shape[0])):
            raise ValueError("Something went wrong with averaging the time series. (hx:%s, hm:%s, l=%d)" % (hx, hm, avg_length))
        # then the matching vectors of the input Xt and old model vector sequence Mk
        # are averaged along the warping function F
        _, _, (x_arr, y_arr) = self.dtw_fn(X, M, dist_only=False)
        averaged_path, positions = np.zeros(len(x_arr)), np.zeros(len(x_arr))
        for p, (i, j) in enumerate(zip(x_arr, y_arr)):
            averaged_path[p] = M[j] + hx * (X[i] - M[j])
            positions[p] = hm * j + hx * i
        # # next we interpolate the distances into the new vector
        interpx = interp1d(positions, averaged_path, kind='linear', assume_sorted=True, fill_value=0.0)
        M_ = interpx(np.linspace(positions[0], positions[-1], num=avg_length))
        if np.isnan(np.dot(M_, M_)):
            raise ValueError("Something went wrong with interpolation, nan-values...")
        # logging.debug("Min / Max value after interpolation: %.4f / %.4f" % (M_.min(), M_.max()))
        return M_ / np.linalg.norm(M_)

    def _init_weights(self, x, y, *args):
        self.weights = [[np.zeros(0) for i in range(x)] for j in range(y)]

    def random_weights_init(self, data):
        """Initializes the weights of the SOM picking random samples from data."""
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            i, j = it.multi_index
            self.weights[i][j] = data[self.random_generator.randint(len(data))].copy()
            it.iternext()

    def quantization_error(self, data):
        """Returns the quantization error computed as the average (normalized) distance between
        each input sample and its best matching unit."""
        error = 0.0
        for x in data:        
            i, j = self.winner(x)
            error += self.dtw_fn(self.weights[i][j], x, normalized=True)
        return error / len(data)

    def distance_map(self):
        """ Returns the average distance map of the weights.
            (Each mean is normalized in order to sum up to 1) """
        um = np.zeros((self.x, self.y))
        for i in range(self.x):
            for j in range(self.y):
                for ii in range(i-1, i+2):
                    for jj in range(j-1, j+2):
                        if ii >= 0 and ii < self.x and jj >= 0 and jj < self.y:
                            um[i, j] += self.dtw_fn(self.weights[i][j], self.weights[ii][jj])
        return um

    def distance_matrix(self):
        return super(DtwSom, self).distance_matrix(dist_fn=self.dtw_fn)
