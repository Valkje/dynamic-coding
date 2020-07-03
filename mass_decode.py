import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output, display
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from sklearn.cluster import KMeans
import multiprocessing
from functools import partial
import time
import os
from os.path import join
import re

import wolff
import wolff_cross

# Track calculation errors
np.seterr('raise')

# start_path = '/Users/share/Chiel4Loran/sweep/exp2'
start_path = '/Users/share/Chiel4Loran/FFxRECxNOISE'
pat_neuro = re.compile(r"neuro_mem")

neuro_prefix = "neuro_mem"
angle_prefix = "initial_angles"

files = []
params = []

for (dirpath, dirnames, filenames) in os.walk(start_path):
    neuro_files = list(filter(pat_neuro.match, filenames))
    _params = [pat_neuro.sub("", file) for file in neuro_files]
    files += [(join(dirpath, neuro_prefix+param), join(dirpath, angle_prefix+param)) for param in _params]
    params += _params
    
params = [re.sub(r"(^_)|(\.npy)", "", param) for param in params]
params_txt = [re.sub(r"_", " ", param) for param in params]
params_txt = [re.sub(r",", ".", param) for param in params_txt]

# Group the thousands of neurons into a handful of channels 
def group(mem_data):
    cut_data = mem_data[:, :500, :] # trials by 500 by neurons
    num_channels = 17
    neurons = np.mean(cut_data, 1).T # neurons by trials
    kmeans = KMeans(n_clusters=num_channels, n_init=20, n_jobs=10, tol=1e-20).fit(neurons)
    
    data = np.empty((mem_data.shape[0], num_channels, mem_data.shape[1])) # trials by num_channels by timesteps
    for channel in range(num_channels):
        print(str(channel + 1) + "/" + str(num_channels), end='\r')
        data[:, channel, :] = np.mean(mem_data[:, :, kmeans.labels_ == channel], axis=2)
    
    return data

base_path = '/Users/s3182541/STSP/Decoding/data/FFxRECxNOISE'
if not os.path.exists(base_path):
    os.mkdir(base_path)

paths_prepared = []

for (dat_file, ang_file), param in zip(files, params):
    print("Preparing for " + param)
#     path = join('/Users/s3182541/STSP/Decoding/data/sweep', param)
    path = join(base_path, param)
    paths_prepared.append(path)
    
    if not os.path.exists(path):
        # Load raw data and angles
        data_raw = np.load(dat_file)
        angles = np.load(ang_file)

        # Prepare data and transform angles
        data = group(data_raw)
        data += np.random.normal(scale=0.5, size=data.shape) # Prevent division by zero errors
        angles = angles / 360 * 2 * np.pi # Convert to radians
        angles = angles * 2 # 'Scale' angles

        if __name__ == '__main__':
            sigma = wolff_cross.prepare_sigma(data)

        # Save all of it to file
        if not os.path.exists(path):
            os.mkdir(path)

        np.save(join(path, "data.npy"), data)
        np.save(join(path, "angles.npy"), angles)
        np.save(join(path, "sigma.npy"), sigma)
        
# Start four processes
# Define a function that:
#  uses a path to read in data 
#  starts the decoding
#  writes the result to disk

def decode_file(path_tup):
    i, path = path_tup
    
    print("Starting with " + path)
    
    if not os.path.exists(join(path, "c.npy")):
        data = np.load(join(path, "data.npy"))
        angles = np.load(join(path, "angles.npy"))
        sigma = np.load(join(path, "sigma.npy"))

        bin_width = np.pi / 6

        device_i = i % 4

        cross_cos_amp = wolff_cross.cross_decode(data, angles, bin_width, sigma, device_i)
        c = np.mean(cross_cos_amp, 0)

        np.save(join(path, "c.npy"), c)
    
    print("Done with " + path)

if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        calcs = pool.imap(decode_file, enumerate(paths_prepared))
        for i, path in enumerate(paths_prepared):
            next(calcs)