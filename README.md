# Dynamic coding

The code associated with the Bachelor's thesis "_Dynamic coding in a large-scale, functional, spiking-neuron model_". This code can be used to analyse (model) data. (See [this repository](https://github.com/ChielWijs/Dynamic-Coding-in-Working-Memory) for a Nengo model that can generate such data.) More specifically, it checks for either the regular or cross-temporal decodability of oriented gratings.

The meat of this code, which can be found in `wolff.py` and `wolff_cross.py`, is based on a paper by Wolff and colleagues (2017), called "_Dynamic hidden states underlying working-memory-guided behavior_".

## `wolff.py`

Code for calculating decodability curves. The file has two main functions: `similarity` and `similarity_p`, where the latter is a parallellised and significantly faster version of the former. They have the following signatures:

```python
def similarity(data, theta, angspace, bin_width):
    ...
    return (cos_amp, distances)

def similarity_p(data, theta, angspace, bin_width, num_cores):
    ...
    return (cos_amp, distances)
```

where the parameters indicate the following:

* `data`: a `trials` by `channels` by `timesteps` numpy array of EEG/model data;
* `theta`: a `trials` long numpy array of associated memory item angles;
* `angspace`: a numpy array of centres of the bins that will be used to bin `data`;
* `binwidth`: a value that indicates the bin width of the aforementioned bins;
* `num_cores`: the maximum number of cores `similarity_p` will use for its calculations.

The returned variables are:

* `cos_amp`: a `trials` by `timesteps` numpy array of decodability values, i.e. a decodability curve for every trial from `data`;
* `distances`: a `trials` by `bins` by `timesteps` numpy array of Mahalanobis distances. Mean-centring, convolving with a cosine and taking the mean across the `bin` dimension gives `cos_amp`.

## `wolff_cross.py`

Code for calculating cross-temporal decodability analyses (CTDAs), designed specifically to use GPUs. The main function to call is:

```python
def cross_decode(data, theta, bin_width, sigma=None, device_i=None):
    ...
    return cross_cos_amp
```

where the parameters indicate the following:

* `data`: a `trials` by `channels` by `timesteps` numpy array of EEG/model data;
* `theta`: a `trials` long numpy array of associated memory item angles;
* `binwidth`: a value that indicates the bin width of the bins. The bin centres (`angspace`) are hardcoded;
* `sigma` (optional): a `trials` by `timesteps` by `channels` by `channels` numpy array of inverse covariance matrices. If not given, will be calculated by `cross_decode`;
* `device_i`: the number of which GPU to use. Not optional: `cross_decode` has been designed specifically for use with a GPU.

The returned variable is:

* `cross_cos_amp`: a `trials` by `timesteps` by `timesteps` array of decodability values, i.e. a CTDA for every trial from `data`.