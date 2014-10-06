import glob
import os

import ConfigParser
import logging
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

from storypy import Story
from dtwsom import DtwSom
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
        characters.append(occurrences.values)
        names.append(actor)

def experiment(args):
    (x, y, sigma, slope, learning_rate, constraint, window, random_seed, step_pattern, iterations, data, training_procedure, initialization_procedure) = args
    som = DtwSom(x, y, sigma=sigma, curve=slope, learning_rate=learning_rate, 
                 random_seed=random_seed, window=window, 
                 step_pattern=step_pattern)
    if initialization_procedure == 'obs':
        som.random_weights_init(data)
    if training_procedure == 'random':
        som.train_random(data, iterations)
    elif training_procedure == 'batch':
        som.train_batch(data, iterations)
    else:
        raise ValueError("Training procedure '%s' is not supported." % training_procedure)
    experiment_name = '%sx%s-sigma.%.2f-slope.%.2f-lr.%.2f-constraint.%s-window.%s-tp.%s-ip.%s' % (
        x, y, sigma, slope, learning_rate, constraint, window, training_procedure, initialization_procedure)
    experiment_name = os.path.join(parser.get("data", "outputdir"), experiment_name)
    som.plot(kind='hinton', data=data, fp=experiment_name + '-hinton.pdf')
    som.plot(kind='grid', what='prototypes', normalize_scale=True, fp=experiment_name + '-proto-grid.pdf')
    som.plot(kind='dmap', fp=experiment_name + '-umatrix.pdf')
    som.plot(kind='grid', what='obs', data=characters, normalize_scale=True, fp=experiment_name + '-obs-grid.pdf')

def parse_range(field):
    if ':' in field:
        start, end = map(float, field.split(':'))
        if start < 1 and end <= 1:
            return np.arange(start, end+0.1, 0.1)
        return range(int(start), int(end)+1)
    return [int(field)]

params = []
for size in parse_range(parser.get("som", "size")):
    size = int(size)
    sigma = parser.get("som", "sigma")
    sigma = size / 3. if sigma == 'relative' else float(sigma)
    for lr in parse_range(parser.get("som", 'learning-rate')):
        for slope in parse_range(parser.get("som", "slope")):
            params.append([size, size, sigma, slope, lr, 
                          parser.get("dtw", "constraint"),
                          int(parser.get("dtw", "window")),
                          int(parser.get("som", "seed")),
                          int(parser.get("dtw", "step-pattern")),
                          int(parser.get("som", "iterations")),
                          characters,
                          parser.get("som", "training-procedure"),
                          parser.get("som", "init-weights")])

n_jobs = int(parser.get('general', 'n-jobs'))
pool = Pool(n_jobs)
pool.map(experiment, params, chunksize=len(params) / n_jobs)
pool.close()
pool.join()