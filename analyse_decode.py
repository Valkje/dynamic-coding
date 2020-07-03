import numpy as np
from scipy.spatial import distance
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

# Track calculation errors
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
        print("Trial " + str(trl+1) + "/" + str(num_trials))
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
            # print("theta[trl]: " + str(theta[trl]))
            # print("trn_angle: " + str(trn_angle))
            # print("angspace[b]: " + str(angspace[b]))
            # print("angle_dists: " + str(angle_dists))
            m[b, :, :] = np.mean(trn_dat[angle_dists < bin_width, : , :], 0)

        for ti in range(0, timesteps):
            print("Trial " + str(trl+1) + "/" + str(num_trials) + "; Ti " + str(ti+1) + "/" + str(timesteps))
            # np.cov expects variables as rows and observations as columns
            # sigma = np.cov(trn_dat[:, :, ti].T)
            # sigma = np.linalg.pinv(sigma)

            sigma = covdiag(trn_dat[:, :, ti])
            sigma = np.linalg.pinv(sigma)
            # Calculate the distances between the trial and all angle bins
            # distances[trl, i, ti] = distance.mahalanobis(m[i, :, ti], data[trl, :, ti], sigma)
            distances[trl, :, ti] = np.array([distance.mahalanobis(means, data[trl, :, ti], sigma) for means in m[:, :, ti]])

            # Convolve cosine of angspace with distances
            # Since a perfect decoding distance curve resembles a reversed
            # cosine (higher distance means higher value), the value is reversed
            # for ease of interpretation, so that higher values mean better
            # decoding
            cos_amp[trl, ti] = -(np.mean(np.cos(angspace) * distances[trl, :, ti].T))

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

    # print(sample)

    # Compute the estimator
    shrinkage = max(0., min(1., r2 / d))
    return shrinkage * prior + (1 - shrinkage) * sample

# Not in use yet, but might provide a speed-up
def mahalanobis(u, v, VI):
    # u.shape: bins by channels
    # v.shape: channels
    # VI.shape: channels by channels

    # delta.shape: bins by channels
    delta = u - v
    # dot(delta, VI).shape: bins by channels
    # delta.T.shape: channels by bins
    # dot(dot(delta, VI), delta.T).shape: bins by bins
    return np.sqrt(np.dot(np.dot(delta, VI), delta.T))

print("Loading data...")

# n trials, 3000 timesteps, 100(0) neurons
sen_data = np.load("data/neuro_sen.npy")

# 100 neurons, n trials, 3000 timesteps
# sen_data = sen_data.transpose(2, 0, 1)
# Mean centre across channels
# sen_data = sen_data - np.mean(sen_data, 0)
# n trials, 100 neurons, 3000 timesteps
# sen_data = sen_data.transpose(1, 0, 2)

# Split neurons into groups of 100
print("Binning neurons...")
sen_data = np.split(sen_data, 10, axis=2) # 10 times n by 3000 by 100
sen_data = np.mean(sen_data, axis=3) # 10 by n by 3000
sen_data = sen_data - np.mean(sen_data, 0) # Mean centre across channels
sen_data = sen_data.transpose(1, 0, 2) # n by 10 by 3000

# Add noise to prevent division by zero errors in covdiag()
sen_data += np.abs(np.random.normal(scale=0.00001, size=(sen_data.shape[0], sen_data.shape[1], sen_data.shape[2])))

print(sen_data.shape)

# n angles, in degrees
angles = np.load("data/initial_angles_cued.npy")
# Convert to radians
angles = angles / 360 * 2 * np.pi

bin_width = np.pi / 6
angspace = np.arange(-np.pi, np.pi, bin_width)
# angspace = np.arange(-np.pi, 0 + bin_width, bin_width) # [-pi, 0] (inclusive)

(cos_amp, distances) = similarity(sen_data, angles, angspace, bin_width)
np.save("data/cos_amp3.npy", cos_amp)
np.save("data/distances3.npy", distances)
