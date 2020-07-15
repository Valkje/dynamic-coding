# Code adapted from Wolff et al. (2017)

import numpy as np
from scipy.spatial import distance
import multiprocessing
from functools import partial

# Throw exceptions on calculation errors
np.seterr('raise')

def similarity(data, theta, angspace, bin_width):
    num_trials = data.shape[0]
    num_channels = data.shape[1]
    timesteps = data.shape[2]

    # distances.shape: trials by bins by time
    distances = np.empty((num_trials, len(angspace), timesteps))
    # cos_amp.shape: trials by time
    cos_amp = np.empty((num_trials, timesteps))

    for trl in range(0, num_trials):
        print("Trial " + str(trl+1) + "/" + str(num_trials), end='\r')
        # Get all data except trl
        trn_dat = data[np.arange(num_trials) != trl, :, :]
        # Get all angles except the one associated with trl
        trn_angle = theta[np.arange(num_trials) != trl]
        # m.shape: bins by channels by time
        m = np.empty((len(angspace), num_channels, timesteps))

        # Average the training data into orientation bins relative to the
        # test-trial's orientation
        for b in range(0, len(angspace)):
            angle_dists = np.abs(np.angle(np.exp(1j*trn_angle) / np.exp(1j*(theta[trl] - angspace[b]))))
            m[b, :, :] = np.mean(trn_dat[angle_dists < bin_width, : , :], 0)

        for ti in range(0, timesteps):
            # Using np.cov gives different results than the Wolff et al. script, covdiag should be used
            sigma = covdiag(trn_dat[:, :, ti])
            sigma = np.linalg.pinv(sigma)
            # Calculate the distances between the trial and all angle bins
            distances[trl, :, ti] = mahalanobis(m[:, :, ti], data[trl, :, ti], sigma)

            # Convolve cosine of angspace with distances
            # Since a perfect decoding distance curve resembles a reversed
            # cosine (higher distance means higher value), the value is reversed
            # for ease of interpretation, so that higher values mean better
            # decoding
            mean_centred = distances[trl, :, ti].T - np.mean(distances[trl, :, ti])
            cos_amp[trl, ti] = -(np.mean(np.cos(angspace) * mean_centred))

    return (cos_amp, distances)

# A parallellised version of similarity that is considerably faster
def similarity_p(data, theta, angspace, bin_width, num_cores):
    num_trials = data.shape[0]
    num_channels = data.shape[1]
    timesteps = data.shape[2]
    
    cos_amp = np.zeros((num_trials, timesteps))
    distances = np.zeros((num_trials, len(angspace), timesteps))
    
    # Apply the function calc_sim partially...
    calc_sim_part = partial(calc_sim, data, theta, angspace, bin_width)
    
    # ...and then run it in parallel
    with multiprocessing.Pool(num_cores) as pool:
        calcs = pool.imap(calc_sim_part, [trl for trl in range(num_trials)], chunksize=10)
        for trl in range(num_trials):
            (cos_amp[trl,], distances[trl,]) = next(calcs)
            print(str(trl+1) + "/" + str(num_trials), end='\r')

    return (cos_amp, distances)

def calc_sim(data, theta, angspace, bin_width, trl):
    num_trials = data.shape[0]
    num_channels = data.shape[1]
    timesteps = data.shape[2]

    # distances.shape: bins by time
    distances = np.empty((len(angspace), timesteps))
    # cos_amp.shape: time
    cos_amp = np.empty(timesteps)
    
    # Get all data except trl
    trn_dat = data[np.arange(num_trials) != trl, :, :]
    # Get all angles except the one associated with trl
    trn_angle = theta[np.arange(num_trials) != trl]
    # m.shape: bins by channels by time
    m = np.empty((len(angspace), num_channels, timesteps))

    # Average the training data into orientation bins relative to the
    # test-trial's orientation
    for b in range(0, len(angspace)):
        angle_dists = np.abs(np.angle(np.exp(1j*trn_angle) / np.exp(1j*(theta[trl] - angspace[b]))))
        m[b, :, :] = np.mean(trn_dat[angle_dists < bin_width, : , :], 0)

    for ti in range(0, timesteps):
        # Using np.cov gives different results than the Wolff et al. script, covdiag should be used
        sigma = covdiag(trn_dat[:, :, ti])
        sigma = np.linalg.pinv(sigma)
        # Calculate the distances between the trial and all angle bins
        distances[:, ti] = mahalanobis(m[:, :, ti], data[trl, :, ti], sigma)

        # Convolve cosine of angspace with distances
        # Since a perfect decoding distance curve resembles a reversed
        # cosine (higher distance means higher value), the value is reversed
        # for ease of interpretation, so that higher values mean better
        # decoding
        mean_centred = distances[:, ti].T - np.mean(distances[:, ti])
        cos_amp[ti] = -(np.mean(np.cos(angspace) * mean_centred))
        
    return (cos_amp, distances)

def covdiag(m):
    # m (t*n): t iid observations on n random variables
    # sigma (n*n): Invertible covariance matrix estimator

    # Subtract column means from every row
    (t, n) = m.shape
    m = m - np.mean(m, axis=0)

    # Compute sample covariance matrix
    sample = (1 / t) * np.matmul(m.T, m)

    # Compute prior
    prior = np.diag(np.diag(sample))

    # Compute shrinkage parameters
    d = 1/n * np.linalg.norm(sample - prior, 'fro') ** 2
    y = m ** 2
    r2 = 1 / n / t ** 2 * np.sum(np.matmul(y.T, y)) - 1 / n / t * np.sum(sample ** 2)

    # Compute the estimator
    shrinkage = max(0., min(1., r2 / d))
    return shrinkage * prior + (1 - shrinkage) * sample

def mahalanobis(u, v, VI):
    # u.shape: bins by channels
    # v.shape: channels
    # VI.shape: channels by channels

    # delta.shape: bins by channels
    delta = u - v
    # dot(delta, VI).shape: bins by channels
    # delta.T.shape: channels by bins
    # dot(dot(delta, VI), delta.T).shape: bins by bins
    return np.sqrt(np.diag(np.dot(np.dot(delta, VI), delta.T)))