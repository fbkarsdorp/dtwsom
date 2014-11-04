import glob
import os

import ConfigParser
import logging

from storypy import Story
from dtwsom import DtwSom
from dgw.dtw.scaling import uniform_scaling_to_length, uniform_shrinking_to_length
import numpy as np
from multiprocessing import Pool

parser = ConfigParser.ConfigParser()
parser.read("dtwexp.config")

verbosity = int(parser.get("general", "verbosity"))
logging_level = logging.INFO if verbosity == 1 else logging.DEBUG if verbosity > 1 else logging.WARN
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging_level)

# read the data into memory
filenames = glob.glob(parser.get("data", "collection"))
stories = map(Story.load, filenames)
characters, names = [], []
for story in stories:
    for actor, occurrences in story.to_dataframe().iterrows():
        if occurrences.max() == 1:
            characters.append(occurrences.values)
            names.append(actor)
chars = []
for char in characters:
    if char.shape[0] < 65:
        chars.append(uniform_scaling_to_length(char, 65))
    elif char.shape[0] > 65:
        chars.append(uniform_shrinking_to_length(char, 65))
    else:
        chars.append(char)
chars = [char / np.linalg.norm(np.array(char)) for char in chars]

def experiment(args):
    (x, y, sigma, sigma_end, eta, eta_end, decay_fn, 
     random_seed, n_iter, constraint, metric, window, 
     step_pattern, normalized, 
     data, training_procedure, initialization_procedure) = args
    
    som = DtwSom(x, y, sigma=sigma, sigma_end=sigma_end, eta=eta, eta_end=eta_end,  
                 decay_fn=decay_fn, random_seed=random_seed, n_iter=n_iter, intermediate_plots=False, 
                 compute_errors=0, constraint=constraint, metric=metric, window=window,
                 step_pattern=step_pattern, normalized=normalized)

    if initialization_procedure == 'obs':
        som.random_weights_init(data)
    if training_procedure == 'random':
        try:
            som.train_random(data)
        except ValueError, err:
            print err
            return
    elif training_procedure == 'batch':
        try:
            som.train_batch(data)
        except ValueError, err:
            print err
            return
    else:
        raise ValueError("Training procedure '%s' is not supported." % training_procedure)
    errors = som.quantization_error(data), som.topology_error(data)
    print errors
    return errors, args

def parse_range(field):
    if ':' in field:
        if len(field.split(':')) == 3:
            start, end, step = map(float, field.split(':'))
            if start < 1 and end <= 1:
                return np.arange(start, end+0.1, step)
            return range(int(start), int(end)+1, int(step))
        else:
            start, end = map(float, field.split(':'))
            if start < 1 and end <= 1:
                return np.arange(start, end+0.1, 0.1)
            return range(int(start), int(end)+1)
    return [int(field)]

params = []
for size in parse_range(parser.get("som", "size")):
    size = int(size)
    sigma = parser.get("som", "sigma")
    sigma = None if sigma == 'relative' else float(sigma)
    for sigma_end in parse_range(parser.get("som", "sigma-end")):
        for eta in parse_range(parser.get("som", 'eta')):
            for eta_end in parse_range(parser.get("som", "eta-end")):
                for window in parse_range(parser.get("dtw", "window")):
                    params.append([size, size, 
                                  sigma, sigma_end,
                                  eta, eta_end, 
                                  parser.get("som", "decay-function"),
                                  int(parser.get("som", "seed")),
                                  int(parser.get("som", "iterations")),
                                  parser.get("dtw", "constraint"),
                                  parser.get("dtw", "metric"),
                                  window,
                                  int(parser.get("dtw", "step-pattern")),
                                  parser.get("dtw", "normalize"),
                                  chars,
                                  parser.get("som", "training-procedure"),
                                  parser.get("som", "init-weights")])

n_jobs = int(parser.get('general', 'n-jobs'))
pool = Pool(n_jobs)
results = pool.map(experiment, params, chunksize=len(params) / n_jobs)
pool.close()
pool.join()
for result in results:
    print result