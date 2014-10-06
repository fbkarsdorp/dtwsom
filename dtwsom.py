import logging
import random
import math

from collections import defaultdict
from functools import partial
from warnings import warn

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from dtw import dtw_distance
from somplot import plot_distance_map, hinton


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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

def resample(ts, values, num_samples):
    """Convert a list of times and a list of values to evenly
    spaced samples with linear interpolation."""
    ts = normalize(ts)
    return normalize(np.interp(np.linspace(0.0, 1.0, num_samples), ts, values))


# the Som class is almost a plain copy of MiniSom 
class Som(object):
    def __init__(self, x, y, input_len=0, sigma=1.0, curve=1, learning_rate=0.5,
                 random_seed=None, intermediate_plots=False):

        """Initializes a Self Organizing Maps.
        x,y - dimensions of the SOM
        input_len - number of the elements of the vectors in input
        sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
        learning_rate - initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
        random_seed, random seed to use."""

        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        self.random_generator = np.random.RandomState(random_seed)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.activation_map = np.zeros((x, y))
        self.curve = curve
        self.x, self.y = x, y
        self.neigx = np.arange(x)
        self.neigy = np.arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian
        self.intermediate_plots = intermediate_plots
        self._init_weights(x, y, input_len)

    def _activate(self,x):
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

    def gaussian(self, c, sigma):
        "Returns a Gaussian centered in c."
        d = 2 * sigma * sigma # * np.pi
        ax = np.exp(-np.power(self.neigx-c[0], 2) / d)
        ay = np.exp(-np.power(self.neigy-c[1], 2) / d)
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
        # eta(t) = eta(0) / (1 + t/T)
        # keeps the learning rate nearly constant for the first T iterations and then adjusts it
        eta = self.learning_rate / (1 + t / self.T)
        sig = self.sigma / (1 + t / self.T) # sigma and learning rate decrease with the same rule
        logging.info('Eta = %.4f / Sigma = %.4f' % (eta, sig))
        g = self.neighborhood(win, sig) * eta # improves the performances
        it = np.nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index] * (x - self.weights[it.multi_index])
            # normalization
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
            self.weights[it.multi_index] = data[int(self.random_generator.rand() * len(data)-1)]
            self.weights[it.multi_index] = self.weights[it.multi_index] / np.linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self, data, num_iteration, debug=False):
        "Trains the SOM picking samples at random from data."
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            if debug:
                self.plot(iteration)
            logging.info('Iteration %d / %d' % (iteration + 1, num_iteration))
            rand_i = int(round(self.random_generator.rand() * len(data)-1))
            input_sequence = data[rand_i] #normalize(data[rand_i])
            self.update(input_sequence, self.winner(input_sequence), iteration)

    def train_batch(self, data, num_iteration, shuffle=False):
        "Trains using all the vectors in data sequentially."
        if shuffle:
            random.shuffle(data)
        self._init_T(len(data))# * num_iteration)
        iteration = 0
        while iteration < num_iteration:
            logging.info('Iteration %d / %d' % (iteration + 1, num_iteration))
            idx = iteration % (len(data) - 1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration):
        "Initializes the parameter T needed to adjust the learning rate."
        self.T = num_iteration / 2.0 # keeps the learning rate nearly constant for the first half of the iterations

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
        dm = np.zeros((len(neurons), len(neurons)))
        for i in range(len(neurons)):
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

    def quantization_error(self, data):
        """Returns the quantization error computed as the average distance between
        each input sample and its best matching unit."""
        error = 0
        for x in data:
            i, j = self.winner(x)
            error += np.linalg.norm(x - self.weights[i][j])
        return error / len(data)

    def win_map(self, data, dist=False):
        """Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
        that have been mapped in the position i,j."""
        winmap = defaultdict(list)
        for x in data:
            i, j = self.winner(x)
            if dist:
                winmap[i, j].append((x, self.activation_map[i, j]))
            else:
                winmap[i, j].append(x)
        return dict(winmap)

    def _cluster(self, method='average', t=0.6):
        dm = self.distance_matrix()
        Z = linkage(squareform(dm), method=method)
        if t is not None:
            return fcluster(Z, t=max(t * Z[:,2]), criterion='distance')
        return Z

    def _plot_grid(self, what='prototypes', data=None, normalize_scale=True):
        fig = plt.figure(figsize=(8, 8))
        outer_grid = gridspec.GridSpec(self.x, self.y, wspace=0.0, hspace=0.0)
        cnt = 0
        width = max(self.weights[i][j].shape[0] for i in range(self.x)
                                                for j in range(self.y))
        if what == 'prototypes':
            points = self.weights
        elif what == 'obs':
            points = [[None for i in range(self.x)] for j in range(self.y)]
            if data is None:
                raise ValueError("Must supply raw data points.")
            wm = self.win_map(data, dist=True)
            for (i, j), value in wm.iteritems():
                points[i][j] = min(value, key=lambda i: i[1])[0]
        for i in range(self.x):
            for j in range(self.y):
                ax = plt.Subplot(fig, outer_grid[cnt])
                if points[i][j] is None:
                    cnt += 1
                    continue
                ax.plot(points[i][j])
                ax.set_xticks([])
                ax.set_yticks([])
                if normalize_scale:
                    ax.set_ylim(0, 1)
                    ax.set_xlim(0, width)
                fig.add_subplot(ax)
                cnt += 1

    def plot(self, t=0, what='prototypes', data=None, kind='grid', normalize_scale=True, fp=""):
        if kind == 'grid':
            self._plot_grid(what=what, data=data, normalize_scale=normalize_scale)
        elif kind == 'dmap':
            plot_distance_map(self.distance_map())
        elif kind == 'hinton':
            if data is None:
                raise ValueError("Must supply raw data points.")
            ar = self.activation_response(data)
            hinton(ar / np.linalg.norm(ar))
        else:
            raise ValueError("Plot type '%s' is not supported." % kind)
        if fp:
            plt.savefig(fp)


class DtwSom(Som):
    def __init__(self, x, y, sigma=1.0, curve=1, learning_rate=0.5,
                 random_seed=None, intermediate_plots=False,
                 constraint='slanted_band', window=5, step_pattern=2,
                 normalized=True):

        super(DtwSom, self).__init__(x, y, sigma=sigma, curve=curve, learning_rate=learning_rate,
                                     random_seed=random_seed, intermediate_plots=intermediate_plots)

        self.dtw_fn = partial(dtw_distance, constraint=constraint,
                                            window=window,
                                            step_pattern=step_pattern,
                                            normalized=normalized)

    def _activate(self,x):
        """Updates matrix activation_map, in this matrix the element i,j
        is the response of the neuron i,j to x """
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            i, j = it.multi_index
            if self.weights[i][j].size == 0:
                # make a new neuron with random signals of a random size
                neuron = np.array(self.random_generator.randint(
                    2, size=self.random_generator.randint(10, 100)), dtype=np.float)
                self.weights[i][j] = normalize(neuron)
            # compute the distance between this neuron and the input 
            self.activation_map[it.multi_index] = self.dtw_fn(self.weights[i][j], x)
            it.iternext()

    def update(self, ts,win,t):
        eta = self.learning_rate * np.exp(-self.curve * ((1.0 + t) / self.T))
        sig = self.sigma * np.exp(-self.curve * ((1.0 + t) / self.T))
        logging.info('Eta = %.4f / Sigma = %.4f' % (eta, sig))
        g = self.neighborhood(win, sig) * eta
        for i in range(self.x):
            for j in range(self.y):
                h = g[i, j]
                logging.debug("Neighborhood function: %.4f / factor %.4f" % (g[i, j], h))
                if h > 0:
                    self.weights[i][j] = self.average_sequence(self.weights[i][j], ts, (1-h), h)
        if self.intermediate_plots and t % 50 == 0:
            self.plot(t)

    def average_sequence(self, M, X, hm, hx):
        """Compute an average sequence between two input sequences based on the warping path
        between them and using linear interpolation for compression."""
        # In the model adaptation the length of the new model vector sequence is determined first
        avg_length = int(0.5 + (hm * M.shape[0] + hx * X.shape[0]))
        if not ((M.shape[0] <= avg_length <= X.shape[0]) or (X.shape[0] <= avg_length <= M.shape[0])):
            raise ValueError("Something went wrong with averaging the time series. (hx:%s, hm:%s, l=%d)" % (hx, hm, avg_length))
        # then the matching vectors of the input Xt and old model vector sequence Mk
        # are averaged along the warping function F
        _, _, (x_arr, y_arr) = self.dtw_fn(X, M, dist_only=False)
        averaged_path, positions = np.zeros(len(x_arr)), np.zeros(len(x_arr))
        for p, (i, j) in enumerate(zip(x_arr, y_arr)):
            averaged_path[p] = M[j] + hx * (X[i] - M[j])
            positions[p] = hm * j + hx * i
        # next we interpolate the distances into the new vector
        M_ = resample(positions, averaged_path, avg_length)
        if not (np.isnan(np.dot(M_, M_)) == False):
            raise ValueError("Something went wrong with interpolation, nan-values...")#(x_arr, y_arr, M_, positions, averaged_path, hm, hx)
        logging.debug("Min / Max value after interpolation: %.4f / %.4f" % (M_.min(), M_.max()))
        return M_

    def _init_weights(self, x, y, *args):
        self.weights = [[np.zeros(0) for i in range(x)] for j in range(y)]

    def random_weights_init(self, data):
        """Initializes the weights of the SOM picking random samples from data."""
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            i, j = it.multi_index
            self.weights[i][j] = data[int(self.random_generator.rand() * len(data) - 1)]
            it.iternext()

    def quantization_error(self,data):
        """Returns the quantization error computed as the average distance between
        each input sample and its best matching unit."""
        error = 0
        for x in data:
            i, j = self.winner(x)
            error += self.dtw_fn(self.weights[i][j], x)
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
