import numpy as np
# import tensorflow as tf
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

import wolff
import wolff_cross

# Track calculation errors
np.seterr('raise')

# tf.debugging.set_log_device_placement(False)

path = '/Users/share/Chiel4Loran/exp2/sim3/'
mem1_file = 'neuro_mem2_1.npy'
mem2_file = 'neuro_mem2_2.npy'
angles_file = 'initial_angles2.npy'

mem_data1 = np.load(path + mem1_file) # trials by timesteps by neurons
mem_data2 = np.load(path + mem2_file)

angles = np.load(path + angles_file)
# Convert to radians
angles = angles / 360 * 2 * np.pi

# 'Scale' angles
angles = angles * 2

def group(mem_data):
    cut_data = mem_data[:, :500, :] # trials by 500 by neurons
    num_channels = 17
    neurons = np.mean(cut_data, 1).T # neurons by trials
    kmeans = KMeans(n_clusters=num_channels, n_init=20, n_jobs=10, tol=1e-20).fit(neurons)
    
    data = np.empty((mem_data.shape[0], num_channels, mem_data.shape[1])) # trials by num_channels by timesteps
    for channel in range(num_channels):
        clear_output(wait=True)
        print(str(channel + 1) + "/" + str(num_channels))
        data[:, channel, :] = np.mean(mem_data[:, :, kmeans.labels_ == channel], axis=2)
        
    return data

data1 = group(mem_data1)
data2 = group(mem_data2)

# Add noise to prevent division by zero errors in covdiag()
data1 += np.random.normal(scale=0.5, size=data1.shape)
data2 += np.random.normal(scale=0.5, size=data2.shape)

bin_width = np.pi / 6
angspace = np.arange(np.pi, np.pi, bin_width)

device_i = 1
instances = 2

if __name__ == '__main__':
    print("Decoding primary...")
    cross_cos_amp1 = wolff_cross.cross_decode(data1, 
                                              angles, 
                                              bin_width, 
                                              device_i=device_i, 
                                              instances=instances)
    print("Decoding secondary...")
    cross_cos_amp2 = wolff_cross.cross_decode(data2, 
                                              angles, 
                                              bin_width, 
                                              device_i=device_i, 
                                              instances=instances)
    
c1 = np.mean(cross_cos_amp1, 0)
c_transformed1 = (c1 + c1.T) / 2

c2 = np.mean(cross_cos_amp2, 0)
c_transformed2 = (c2 + c2.T) / 2

np.save('c_transformed1.npy', c_transformed1)
np.save('c_transformed2.npy', c_transformed2)