# single epoch signal augmentation methods

import numpy as np
import math 
import tsaug
from scipy import signal

def add_noise(epoch, noise_scale=0.01):
    """
    Add noise to EEG epoch each channel
    Args:
        epoch: a epoch of the whole epochs from an EEG recording
        noise_scale: the amount of noise add to epoch signals
    Returns: 
        augmented numpy array
    """
    noise_scales = [0.02, 0.05, 0.08, 0.1]
    nosie_idx =  np.random.randint(len(noise_scales))
    
    return np.array(tsaug.AddNoise(scale=noise_scales[nosie_idx]).augment(epoch))


def downsample(epoch, Hz=128, epoch_length=6):
    """ 
    Downsample the EEG epoch to Hz=128 Hz and to only
    include the channels in ch.
    Args:
        epoch: a epoch of the whole epochs from an EEG recording
        Hz: Hz to downsample to 
        epoch_length: the length of epoch in seconds
    Returns:
        augmented numpy array
    """
    HZs = [256, 200, 160, 128, 100]
    Hz_idx = np.random.randint(len(HZs))
    
    return np.array(signal.resample(epoch, HZs[Hz_idx]*epoch_length, axis = 1)) # epoch has the shape [channel, time_points]

def random_select_channels(epoch, num_channels=19):
    """
    Randomly select number of channels from epoch
    Args:
        epoch: a epoch of the whole epochs from an EEG recording
        num_channels: number of channels to select from total channels
    Return:
        augmented numpy array with random selected number of channels
    """
    #Not right. In this way, it may selection duplicated channels and disorder the channels
    #TO DO: find a better way
    idx = np.random.randint(len(epoch), size=num_channels)
    return np.array(epoch[idx, :])

def random_clip(epoch, clip_length=1000):
    """
    Randomly clip epoch to the clip_length
    Args:
        epoch: a epoch of the whole epochs from an EEG recording
        clip_length: length of continues epoch signals clipped from original epoch
    Return:
        augmented numpy array with length = clip_length 
    """
    epoch_length = len(epoch[0])
    
    clip_lengths = [600, 700, 800, 900, 1000, 1100, 1200]
    clip_idx = np.random.randint(len(clip_lengths))
    
    clip_length = clip_lengths[clip_idx]
    if clip_length > len(epoch[0]):
        raise ValueError('clip_length must smaller than original signal length')
    
    start_idx = np.random.randint(epoch_length - clip_length - 1)
    end_idx = min(start_idx+clip_length+1, epoch_length) 
    
    epoch = epoch[:, start_idx:end_idx]
    
    return epoch


def pool(epoch, pool_size=2):
    """
    Reduce the temporal resolution without changing the length.
    Args:
        epoch: a epoch of the whole epochs from an EEG recording
        pool_size: pooling size the resude resolution of input epoch
    Return:
        augmented numpy array with the same shape of epoch
    """
    pool_sizes = [2, 3, 4, 5]
    pool_idx = np.random.randint(len(pool_sizes))
    
    return np.array(tsaug.Pool(size=pool_sizes[pool_idx]).augment(epoch))

def quantize(epoch, quantize_level=20):
    """
    Quantize time series to a level set.
    Args:
        epoch: a epoch of the whole epochs from an EEG recording
        quantize_level: Values in a time series are rounded to the nearest level in the level set.
    """
    quantize_levels = [10, 15, 20, 25, 30]
    quantize_idx = np.random.randint(len(quantize_levels))
    
    return np.array(tsaug.Quantize(n_levels=quantize_levels[quantize_idx]).augment(epoch))
    