# This file load the TUH Normal / Abnormal dataset

import random

import os
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from math import floor, ceil


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#from ssl.new_SSL_TS_RP import temporal_shuffling, relative_positioning
from preprocessing.preprocess import preprocess
from preprocessing.signalTransformation import *

MINUTES_TO_SECONDS = 60


'''
For TUH normal / abnormal dataset, 
the author use the eval set as test set. 
The train set is divided as 80% - 20% as train and evaluation set. 
In the paper: 
    400 pairs for relative positioning and temporal shifting pairs. 
    
Preprocessing: 
1. Cropped the first minute recordings.
2. Cropped the recording length to < 20 minutes. 
3. Select 19 channels that are common to all recordings are selected:
    Fp1, Fp2, F7, F8, F3, Fz, F4, 
    T3, C3, Cz, C4, T4, T5, 
    P3, Pz, P4, T6, O1, O2
4. Downsample to 100Hz, clipped at +-800uV
5. 6 seconds non-overlappig window are extracted. Each window size is 600 x 21 
7. peak-to-peak amplitude below 1uV are rejected. 
8. normalized channel-wise to have zero-mean and unit standard deviation
'''


class Normal_Dataset(Dataset):
    #TO DO: Find proper anchor_windows_per_recording and samples_per_anchor_window
    #TO DO: Find proper T_pos and T_neg
    def __init__(self, T_neg = 10, samples_per_recording=200,
    raw_data_files=None, preprocessed_file=None,
    save_preprocessed_path=None, window_length=6):
        """
        Takes in either a data folder or a preprocessed file
        """
        self.window_length = window_length # in seconds
        self.samples_per_recording = samples_per_recording
        self.T_neg = T_neg # The negative time window should be sampled outside 10 idx away from the anchor window

        if ((raw_data_files is None and preprocessed_file is None)
            or (raw_data_files is not None and preprocessed_file is not None)):
            raise ValueError("Dataset requires a preprocessed_file or a raw_data_files")
        
        if raw_data_files is not None:
            f = open(raw_data_files, "r")
            files = []
            while True:
                line = f.readline()
                if not line:
                    break
                files.append(line.strip())
            f.close()
            self.files = files
            
            self.preprocessed = []
            for f in tqdm(self.files):
                pp_file = preprocess(f)
                self.preprocessed.append(pp_file)
            
            #TO DO: need to save all pre-processed file to reduce data pre-processing time
            if save_preprocessed_path is not None:
                pickle.dump((self.preprocessed, self.files), open(save_preprocessed_path, 'wb'))

        elif preprocessed_file is not None:
            self.preprocessed, self.files = pickle.load(open(preprocessed_file, 'rb'))

        self.num_files = len(self.files)

    def __len__(self):
        return self.num_files * self.samples_per_recording  # Total number of samples 

    def __getitem__(self, idx):
        
        file_idx = idx % self.num_files
        f = self.preprocessed[file_idx]
        pos_neg_flag = np.random.randint(2) #Generate random 0 & 1
        anchor_idx = np.random.randint(len(f))

        ### Sampling with the indexes
        org_window, aug_window, aug_label = self.random_augmentation(f, anchor_idx, pos_neg_flag)
        
        #change data shape from [ch, length] to [1, ch, length]
        org_window = np.expand_dims(org_window, axis=0)
        aug_window = np.expand_dims(aug_window, axis=0)
        
        org_window = torch.tensor(org_window)
        aug_window = torch.tensor(aug_window)
        aug_label = torch.tensor(aug_label)
        
        return org_window, aug_window, aug_label
    
    def collate_fn(self, batch):
        org_windows = []
        org_windows_ch = []
        org_windows_length = []
        
        aug_windows = []
        aug_windows_ch = []
        aug_windows_length = []
        
        labels = []
        
        for b in batch:
            org_windows.append(b[0])
            org_windows_ch.append(b[0].shape[1])
            org_windows_length.append(b[0].shape[2])
            
            aug_windows.append(b[1])
            aug_windows_ch.append(b[1].shape[1])
            aug_windows_length.append(b[1].shape[2])
            labels.append(b[2])
        
        org_max_ch = max(org_windows_ch)
        org_max_length = max(org_windows_length)
        
        aug_max_ch = max(aug_windows_ch)
        aug_max_length = max(aug_windows_length)
        
        
        org_windows_padded = []
        aug_windows_padded = []
        #padding
        for org_window, ch, length in zip(org_windows, org_windows_ch, org_windows_length):
            pad = nn.ZeroPad2d((0, org_max_length - length, 0, org_max_ch - ch))
            org_windows_padded.append(pad(org_window))
        
        for aug_window, ch, length in zip(aug_windows, aug_windows_ch, aug_windows_length):
            pad = nn.ZeroPad2d((0, aug_max_length - length, 0, aug_max_ch - ch))
            aug_windows_padded.append(pad(aug_window))
        
        org_windows_padded = torch.stack(org_windows_padded)
        aug_windows_padded = torch.stack(aug_windows_padded)
        
        labels = torch.stack(labels)
        #BCEWITHLOGITSLOSS require labels to be float tensor
        #labels.float()
        
        return org_windows_padded, aug_windows_padded, labels
    
    
    def sample_pos_aug(self, anchor_window):
        """
        Generate an positive augmentation window of anchor_window
        """
        #augment_list = ['add_noise', 'downsample', 'random_select_channels', 'random_clip', 'pool', 'quantize']
        augment_list = ['add_noise', 'downsample', 'random_clip', 'pool', 'quantize']
        augment_idx = np.random.randint(len(augment_list))
        
        # TO DO: apply multiply augmentation in one anchor_window
        if augment_idx == 0:
            aug_window = add_noise(anchor_window)
        elif augment_idx == 1:
            aug_window = downsample(anchor_window, epoch_length=self.window_length)
#         elif augment_idx == 2:
#             aug_window = random_select_channels(anchor_window, num_channels=21)
        elif augment_idx == 2:
            aug_window = random_clip(anchor_window)
        elif augment_idx == 3:
            aug_window = pool(anchor_window)
        elif augment_idx == 4:
            aug_window = quantize(anchor_window)
        else:
            raise ValueError("augmentation method not listed")
            
        return aug_window
    
    
    def sample_neg_aug(self, neg_window):
        #augment_list = ['add_noise', 'downsample', 'random_select_channels', 'random_clip', 'pool', 'quantize', 'original']
        augment_list = ['add_noise', 'downsample', 'random_clip', 'pool', 'quantize', 'original']
        augment_idx = np.random.randint(len(augment_list))
        
        if augment_idx == 0:
            aug_window = add_noise(neg_window)
        elif augment_idx == 1:
            aug_window = downsample(neg_window, epoch_length=self.window_length)
#         elif augment_idx == 2:
#             aug_window = random_select_channels(neg_window, num_channels=21)
        elif augment_idx == 2:
            aug_window = random_clip(neg_window)
        elif augment_idx == 3:
            aug_window = pool(neg_window)
        elif augment_idx == 4:
            aug_window = quantize(neg_window)
        elif augment_idx == 5:
            aug_window = neg_window
        else:
            raise ValueError("augmentation method not listed")
            
        return aug_window
    
    def random_augmentation(self, recording, anchor_idx, pos_neg_flag):
        """
        Retrives a self-supervised data augmentation sample 
        Args:
            recording: numpy dataset of time-seris arrays
            anchor_idx: random selected epoch idx in recording
            pos_neg_flag: random 0 and 1 to generate negative and positive pairs 
        """
        
        anchor_window = recording[anchor_idx]
        #choose positive pairs
        if pos_neg_flag == 1:
            # generate a augmented sample from anchor_dex
            pos_window = self.sample_pos_aug(anchor_window)
            #BCEWithLogitsLoss requires float type label
            aug_label = np.array([1.])
            aug_window = pos_window
        else:
            #generate a random augmented sample from other potion of the recording 
            #The neg_idx should not be within 20 idxs of anchor_idx
            #Therefore the neg_window signal should be a lot different from pos_window
            neg_idx = anchor_idx
            while anchor_idx == neg_idx or abs(anchor_idx - neg_idx) < self.T_neg:
                neg_idx = np.random.randint(len(recording))
            neg_window = self.sample_neg_aug(recording[neg_idx])
            aug_label = np.array([0.])
            aug_window = neg_window
            
        return anchor_window, aug_window, aug_label
    
    
