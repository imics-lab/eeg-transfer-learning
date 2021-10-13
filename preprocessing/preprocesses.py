import os
import numpy as np
import mne
from mne import preprocessing
import sys
import time
from scipy import signal

'''
Preprocessing: 
1. Cropped the first minute recordings.
2. Cropped the recording length to < 20 minutes. 
3. Select 19 channels that are common to all recordings are selected:
    Fp1, Fp2, F7, F8, F3, Fz, F4, 
    T3, C3, Cz, C4, T4, T5, 
    P3, Pz, P4, T6, O1, O2
4. Downsample to 100Hz, clipped at +-800uV  (how to clip？！)
5. 6 seconds non-overlappig window are extracted. Each window size is 600 x 21 
7. peak-to-peak amplitude below 1uV are rejected. (how to?！) 
8. normalized channel-wise to have zero-mean and unit standard deviation
'''



def preprocess(f):
    """ Runs the whole pipeline and returns NumPy data array"""
    epoch_length = 6 # s
    #TO DO: find the right channels to pick
    CHANNELS = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
                'EEG F7-REF', 'EEG F8-REF', 'EEG FZ-REF',
                'EEG A1-REF', 'EEG A2-REF',
                'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
                'EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF',
                'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF',
                'EEG O1-REF', 'EEG O2-REF'
               ]
    
    _1020_channels = ['FP1', 'FP2', 
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T3', 'C3', 'CZ', 'C4', 'T4',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                  'O1', 'O2']
    
    raw = mne.io.read_raw_edf(f, preload=True, verbose='ERROR')
    
    raw.crop(tmin = 60, tmax = min((raw.n_times - 1)/raw.info['sfreq'], 1200))
    
#     #reomove channels
#     raw = remove_channels(raw, CHANNELS)
    
    #rename and reorder raw channels
    raw.rename_channels(lambda x: x.replace('EEG ', '').replace('-REF', ''))
    raw.pick_channels(_1020_channels)
    raw.reorder_channels(_1020_channels)    
    if len(raw.ch_names) != 19:
        raise ValueError('too many or too less channels are slected')
    
    #add band pass filter
    mne_filtered = filter_eeg(raw, raw.ch_names)
    
    epochs = divide_epochs(mne_filtered, epoch_length)
    
    #convert data to microvoltage
    epochs = epochs * 1e6  
    
    #TO DO: find ways to remove very big voltage channels to remove some artifacts
    
    
    #Make every inputs to the same smapling rate 
    #epochs = downsample(epochs, Hz=100, epoch_length = 6) 

    # normalize epochs along each channel 
    f_epochs = normalization(epochs) # should update this
    
    
    # TO DO: save files to speedup pre-processing process
    #np.save(file[:file.index("-")], f_epochs)
    
    return f_epochs


def remove_channels(mne_raw, channels):
    """Extracts CHANNELS channels from MNE_RAW data.
    Args:
    raw - mne data strucutre of n number of recordings and t seconds each
    channels - channels wished to be extracted
    Returns:
    extracted - mne data structure with only specified channels
    """
    extracted = mne_raw.pick_channels(channels)
    return extracted

def filter_eeg(mne_eeg, channels):
    """Creates a (0.3-80Hz) fifth-order band-pass butterworth filter that is applied to the channels channels from the MNE_EEG data.
    Args:
        mne-eeg - mne data strucutre of n number of recordings and t seconds each
    Returns:
        filtered - mne data structure after the filter has been applied
    """
    # TO DO: find the proper filter we need to apply (doesn't claimed on the paper)
    filtered = mne_eeg.filter(l_freq=0.3,
            h_freq= 80,
            picks = channels,
            filter_length = "auto",
            method = "fir",
            verbose='ERROR'
            )
    return filtered

def divide_epochs(raw, epoch_length):
    """ Divides the mne dataset into many samples of length epoch_length seconds.
    Args:
        E: mne data structure
        epoch_length: (int seconds) length of each sample
    Returns:
        epochs: mne data structure of (experiment length * users) / epoch_length
    """
    # TO DO: Figure out the proper epochs devided length 
    raw_np = raw.get_data()
    s_freq = int(raw.info['sfreq'])
    n_channels, n_time_points = raw_np.shape[0], raw_np.shape[1]

    # make n_time_points a multiple of epoch_length*s
    chopped_n_time_points = n_time_points - (n_time_points % (epoch_length*s_freq))
    raw_np = raw_np[:,:chopped_n_time_points]

    return raw_np.reshape(n_channels, -1, epoch_length*s_freq).transpose(1,0,2)

def downsample(epochs, Hz=100, epoch_length = 6):
    """ Downsample the EEG epoch to Hz=128 Hz and to only
        include the channels in ch.
        Args:
            epochs: mne data structure sampled at a rate r’ > 128 Hz
            chs: list of the channels to keep
            Hz: Hz to downsample to (default 128 Hz)
        Returns
            E: a mne data structure sampled at a rate r of 128 Hz.
    """
    #E = epochs.pick_types(eeg=True, selection=chs, verbose='ERROR')
    #E = E.resample(Hz, npad='auto')
    E = signal.resample(epochs, Hz*epoch_length, axis = 2)
    return E

def _normalize(epoch):
    """ A helper method for the normalization method.
        Args:
            epochs: mne data structure sampled at a rate r’ > 128 Hz
        Returns
            result: a normalized epoch
    """
    # TO DO: find a proper way to normailize the input signals 
    result = (epoch - epoch.mean(axis=0)) / (np.sqrt(epoch.var(axis=0)))
    #print(f'epoch.var = {epoch.var(axis=0)}') # some channel has var = 0
    return result

def normalization(epochs):
    """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
        Args:
            epochs - Numpy structure of epochs
        Returns:
            epochs_n - mne data structure of normalized epochs (mean=0, var=1)
    """
    for i in range(epochs.shape[0]): # TODO could switch to a 1-line numpy matrix operation
        for j in range(epochs.shape[1]):
            epochs[i,j,:] = _normalize(epochs[i,j,:])

    return epochs

def save(f_epochs, name, output_folder):
    """ Saves each epoch as a file
    Args:
    f_epochs - Numpy structure of epochs
    name - file name based on its original mne file name
    output_folder - folder name where data should be saved
    """
    np.save(output_folder + "/epoch_{num}.npy".format(num = name), f_epochs)

if __name__ == '__main__':
    raw_data_files = sys.argv[1]
    preprocess(raw_data_files)
