# Code adapted from Wolff et al. (2017)

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from sklearn.cluster import KMeans
import multiprocessing
from functools import partial
import os

import wolff

# Track calculation errors
np.seterr('raise')

# A helper function to prepare_sigma
def cov_mats(data, trl):
    num_trials = data.shape[0]
    num_channels = data.shape[1]
    timesteps = data.shape[2]
    
    trn_dat = data[np.arange(num_trials) != trl, :, :]
    sigma = np.empty((timesteps, num_channels, num_channels))
    for ti in range(timesteps):
        # Calculate inverse covariance matrix for all data except data[trl] at time step ti
        sigma[ti, :, :] = np.linalg.pinv(wolff.covdiag(trn_dat[:, :, ti]))

    return sigma
    
# Prepare all the inverse covariance matrices needed in the CTDA
def prepare_sigma(data):
    num_trials = data.shape[0]
    num_channels = data.shape[1]
    timesteps = data.shape[2]
    sigma = np.empty((num_trials, timesteps, num_channels, num_channels))
    
    cov_mats_part = partial(cov_mats, data)
    
    # Fill sigma in a parallel fashion
    with multiprocessing.Pool(20) as pool:
        print("Preparing sigma...")
        
        calcs = pool.imap(cov_mats_part, [trl for trl in range(num_trials)], chunksize=10)
        for trl in range(num_trials):
            sigma[trl, :, :, :] = next(calcs)
            print(str(trl+1) + "/" + str(num_trials), end='\r')
            
        print("Done with sigma.")
                
    return sigma

# Calculate a CTDA with GPU device_i
def cross_decode(data, theta, bin_width, sigma=None, device_i=None):
    if sigma is None:
        sigma = prepare_sigma(data)
        
    if device_i is None:
        raise TypeError("device_i has to be specified")
    
    num_trials, num_channels, timesteps = data.shape
    cross_cos_amp = np.empty((num_trials, timesteps, timesteps))
    
    # Limit the visible GPUs to GPU device_i
    os.environ["CUDA_VISIBLE_DEVICES"]=str(device_i)
    
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(False)
    
    with tf.device('/GPU:0'):
        delta = tf.Variable(tf.zeros([timesteps, num_channels], dtype='float64'))

        @tf.function
        def mahalanobis(u, v, VI):
            # u.shape: timesteps by channels
            # v.shape: channels
            # VI.shape: channels by channels

            # delta.shape: timesteps by channels
            delta = u - v

            # dot(delta, VI).shape: timesteps by channels
            # delta.T.shape: channels by timesteps
            # dot(dot(delta, VI), delta.T).shape: timesteps by timesteps
            return tf.sqrt(tf.linalg.diag_part(tf.matmul(delta @ VI, delta, transpose_b=True)))

        # Calculate time steps Mahalanobis distances for all time steps in parallel
        @tf.function
        def parallel(data, m, sigma):
            # data: Transposed data except data[trl]
            # m: bin data
            calc_dist_part = lambda x: mahalanobis(data, x[0], x[1])
            return tf.map_fn(calc_dist_part, 
                             (m, sigma), 
                             dtype=tf.float64, 
                             parallel_iterations=100)

        angspace = np.arange(-np.pi, np.pi, bin_width)
        cosines = np.expand_dims(np.cos(angspace), (1, 2))
        distances = np.empty((len(angspace), timesteps, timesteps))
        
        # Calculate a CTDA for the trial trl
        def amps(trl):
            trn_dat = data[np.arange(num_trials) != trl, :, :]
            trn_angle = theta[np.arange(num_trials) != trl]
            for b in range(len(angspace)):
                angle_dists = np.abs(np.angle(np.exp(1j*trn_angle) / np.exp(1j*(theta[trl] - angspace[b]))))
                m = np.mean(trn_dat[angle_dists < bin_width, : , :], 0)

                # A time steps by time steps grid of decodability scores for bin b
                distances[b] = parallel(tf.constant(data[trl].T), tf.constant(m.T), tf.constant(sigma[trl]))

            # Mean centre all decodability grids, convolve them with a cosine and average them to get a CTDA
            mean_centred = distances - np.mean(distances, 0)
            return -np.mean(cosines * mean_centred, 0)
        
        for trl in range(num_trials):
            print("Trial " + str(trl+1) + "/" + str(num_trials))
            cross_cos_amp[trl, :, :] = amps(trl)
            
    return cross_cos_amp
    