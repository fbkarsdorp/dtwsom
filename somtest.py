import glob
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)
from storypy import Story
from dtwsom import DtwSom
import numpy as np

from dgw.dtw.scaling import uniform_scaling_to_length, uniform_shrinking_to_length


filenames = glob.glob("/Users/folgert/Dropbox/stories/data/sinnighe-actor/SINVS*.txt")
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
for i in range(10):
    som = DtwSom(6, 6, constraint='slanted_band', window=5, sigma=None, 
                 sigma_end=0.5, eta=0.5, eta_end=0.01, decay_fn='exponential', 
                 random_seed=2, metric='euclidean', normalized=True, 
                 step_pattern=2, n_iter=100, compute_errors=0)
    som.train_random(chars)
    print som.topology_error(chars)
