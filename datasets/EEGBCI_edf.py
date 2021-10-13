# This file load the EEGBCI dataset 

import numpy as np
from tqdm import tqdm

# from preprocessing.preprocess import preprocess
from torch.utils.data import Dataset, DataLoader

# mne imports
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

import torch

class EEGBCI_Dataset(Dataset):
    def __init__(self, raw_data_path='/home/x_l30/EEG_analysis/datasets/EEGBCI_dataset', task_number = 1):
        
        self.raw_data_path = raw_data_path
        
        task_runs = {1:[3, 7, 11], 2:[4, 8, 12], 3:[5, 9, 13], 4:[6, 10, 14]}
        self.runs = task_runs[task_number]
        self.event_id = dict(T1=1, T2=2)
        
        if raw_data_path is None:
            raise ValueError("Need EEGBCI dataset data path")
        
#         # remove some subjects 
#         subject1 = [i for i in range(1, 51)]
#         subject2 = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
#                      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 
#                      71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
#                      81, 82, 83, 84, 85, 86, 87,     89, 90, 
#                      91,     93, 94, 95, 96, 97, 98, 99,     
#                      101, 102, 103, 104, 105, 106, 107, 108, 109]
#         self.subjects = subject1 + subject2
        self.subjects = [i for i in range(1, 51)]
#         self.subjects = [1, 2, 3]
        
        self.preprocessed = []
        self.labels = []
        for subject in self.subjects:
            raw_fnames = eegbci.load_data(subject, self.runs, raw_data_path)
            subject_raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
            pp_epochs, labels = self.preprocess(subject_raw, self.event_id)
            self.preprocessed.extend(pp_epochs)
            self.labels.extend(labels)
        
        
        #expend one dimention 
        self.preprocessed = np.array(self.preprocessed)
        self.preprocessed = self.preprocessed.reshape(self.preprocessed.shape[0], 1, self.preprocessed.shape[1], self.preprocessed.shape[2])
        self.preprocessed = torch.tensor(self.preprocessed)
        
        self.labels = np.array(self.labels)
        self.labels = self.labels.reshape(self.labels.shape[0], 1)
#         self.labels = torch.tensor(self.labels).to(torch.float)
        
        print(self.preprocessed.shape)
        print(self.labels.shape)
        
    def preprocess(self, raw, event_id):
        
        _1005_mapping = ['FP1', 'FP2', 
                'F7', 'F3', 'FZ', 'F4', 'F8',
                'T7', 'C3', 'CZ', 'C4', 'T8',
                'P7', 'P3', 'PZ', 'P4', 'P8',
                          'O1', 'O2']
        
        _1020_channels = ['FP1', 'FP2', 
                'F7', 'F3', 'FZ', 'F4', 'F8',
                'T3', 'C3', 'CZ', 'C4', 'T4',
                'T5', 'P3', 'PZ', 'P4', 'T6',
                          'O1', 'O2']
                                           
        tmin, tmax = -1., 4.                                    
        eegbci.standardize(raw)
        # set channel names standard_1005
        montage = make_standard_montage('standard_1005') # Electrodes are named and positioned according to the international 10-05 system (343+3 locations)
        raw.set_montage(montage)

        #strip channel names of "." characters
        raw.rename_channels(lambda x: x.strip('.'))
        raw.rename_channels(lambda x: x.upper())
        raw = raw.pick_channels(_1005_mapping)
        raw.rename_channels({'T7':'T3'})
        raw.rename_channels({'T8':'T4'})
        raw.rename_channels({'P7':'T5'})
        raw.rename_channels({'P8':'T6'})
        raw.reorder_channels(_1020_channels)    
        if len(raw.ch_names) != 19:
            raise ValueError('too many or too less channels are slected')
        

        #Apply band-pass filter 
        raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')  # band pass filter 

        events, _ = events_from_annotations(raw, event_id=event_id)

        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                        exclude='bads')

        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
        
        labels = epochs.events[:, -1] - 1

        #change to microvoltage
        epochs = epochs.get_data() * 1e6
        
        epochs = self.normalization(epochs)
        
        return epochs, labels
    
    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Args:
                epochs: mne data structure sampled at a rate r’ > 128 Hz
            Returns
                result: a normalized epoch
        """

        result = (epoch - epoch.mean(axis=0)) / (np.sqrt(epoch.var(axis=0)))
        #TO DO: remove some EEGBCI channels
        if epoch.var(axis = 0) == 0:
            raise ValueError('epoch channel variance can not be 0')
        return result

    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]): # TODO could switch to a 1-line numpy matrix operation
            for j in range(epochs.shape[1]):
                epochs[i,j,:] = self._normalize(epochs[i,j,:])

        return epochs
                                           
    def __len__(self):
                                           
        return len(self.labels)
                                
    def __getitem__(self, idx):
        return self.preprocessed[idx], self.labels[idx]