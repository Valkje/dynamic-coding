import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from sklearn.cluster import KMeans
import multiprocessing
from functools import partial
import time

import wolff

# Track calculation errors
np.seterr('raise')

def cov_mats(data, trl):
    num_trials = data.shape[0]
    num_channels = data.shape[1]
    timesteps = data.shape[2]
    
    trn_dat = data[np.arange(num_trials) != trl, :, :]
    sigma = np.empty((timesteps, num_channels, num_channels))
    for ti in range(timesteps):
        sigma[ti, :, :] = np.linalg.pinv(wolff.covdiag(trn_dat[:, :, ti]))

    return sigma
    
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
                
def wrap(data, theta, bin_width, sigma, device_i, trl):
    import tensorflow as tf

    num_trials = data.shape[0]
    num_channels = data.shape[1]
    timesteps = data.shape[2]
    
    tf.config.experimental_run_functions_eagerly(False)
    if device_i is None:
        device = '/GPU:' + str(trl % 4)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[device_i], 'GPU')
        device = '/GPU:0'

    with tf.device(device):
        delta = tf.Variable(tf.zeros([timesteps, num_channels], dtype='float64'))

#         @tf.function
#         def mahalanobis(u, v, VI):
#             # u.shape: timesteps by channels
#             # v.shape: channels
#             # VI.shape: channels by channels

#             # delta.shape: timesteps by channels
#             delta = u - v

#             # dot(delta, VI).shape: timesteps by channels
#             # delta.T.shape: channels by timesteps
#             # dot(dot(delta, VI), delta.T).shape: timesteps by timesteps
#             return tf.sqrt(tf.linalg.diag_part(tf.matmul(delta @ VI, delta, transpose_b=True)))

        @tf.function
        def mahalanobis(u, v, VI):
            # u.shape: timesteps by channels
            # v.shape: channels
            # VI.shape: channels by channels

            # delta.shape: timesteps by channels
            delta = u - v
            
            func = lambda x: x[None] @ VI @ tf.transpose(x[None])

            # delta[ti] @ VI: channels
            # (delta[ti] @ VI) @ delta[ti].T: scalar
            return tf.squeeze(tf.vectorized_map(func, delta))

        @tf.function
        def parallel(data, m, sigma):
            calc_dist_part = lambda x: mahalanobis(data, x[0], x[1])
#             return tf.map_fn(calc_dist_part, 
#                              (m, sigma), 
#                              dtype=tf.float64, 
#                              parallel_iterations=100)
            return tf.vectorized_map(calc_dist_part, (m, sigma))
        
        def amps(trl):
            angspace = np.arange(-np.pi, np.pi, bin_width)
            cosines = np.expand_dims(np.cos(angspace), (1, 2))
            distances = np.empty((len(angspace), timesteps, timesteps))

            trn_dat = data[np.arange(num_trials) != trl, :, :]
            trn_angle = theta[np.arange(num_trials) != trl]
            for b in range(len(angspace)):
                angle_dists = np.abs(np.angle(np.exp(1j*trn_angle) / np.exp(1j*(theta[trl] - angspace[b]))))
                m = np.mean(trn_dat[angle_dists < bin_width, : , :], 0)

                distances[b] = parallel(tf.constant(data[trl].T), tf.constant(m.T), tf.constant(sigma[trl]))

            mean_centred = distances - np.mean(distances, 0)
            return -np.mean(cosines * mean_centred, 0)

        return amps(trl)
    
def parallel_decode(wrap_part, num_trials, timesteps, instances):
    cross_cos_amp = np.empty((num_trials, timesteps, timesteps))
    
    with multiprocessing.Pool(instances) as pool:
        calcs = pool.imap(wrap_part, [trl for trl in range(num_trials)])
        for trl in range(num_trials):
            print("Trial " + str(trl+1) + "/" + str(num_trials), end='\r')
            cross_cos_amp[trl, :, :] = next(calcs)
                
    return cross_cos_amp

def cross_decode(data, angles, bin_width, sigma=None, device_i=None, instances=10):
    if sigma is None:
        sigma = prepare_sigma(data)
    
    partial_wrap = partial(wrap, data, angles, bin_width, sigma, device_i)
    return parallel_decode(partial_wrap, data.shape[0], data.shape[2], instances)