import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

cos_amp = np.load("data/cos_amp3.npy")
cos_amp = gaussian_filter(np.mean(cos_amp, axis=0), sigma=8)
plt.plot(cos_amp)
# plt.plot(np.mean(cos_amp, axis=0))
# for i in range(10):
#     plt.plot(cos_amp[i, :])
plt.savefig('figures/cos_amp_mean.eps', format='eps', dpi=1000)
plt.show()
