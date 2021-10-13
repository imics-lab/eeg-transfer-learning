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


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from .tuh_preprocess import *

MINUTES_TO_SECONDS = 60


class TUH_Normal_Abnormal(Dataset):
    def __init__(self, normal_filename=None, abnormal_filename = None, window_length=6):
        """
        Takes in either a data folder or a preprocessed file
        """
        self.window_length = window_length # in seconds
        
        
        normal_files = []
        if normal_filename is not None:
            f = open(normal_filename, "r")
            normal_files = []
            while True:
                line = f.readline()
                if not line:
                    break
                normal_files.append(line.strip())
            f.close()
            
            normal_preprocessed = []
            normal_labels = []
            for f in tqdm(normal_files):
                pp_file = preprocess(f)
                normal_preprocessed.extend(pp_file)
                normal_labels.extend([[1]] * len(pp_file))
            
        
        
        abnormal_files = []
        if abnormal_filename is not None:
            f = open(abnormal_filename, "r")
            abnormal_files = []
            while True:
                line = f.readline()
                if not line:
                    break
                abnormal_files.append(line.strip())
            f.close()
            
            abnormal_preprocessed = []
            abnormal_labels = []
            for f in tqdm(abnormal_files):
                pp_file = preprocess(f)
                abnormal_preprocessed.extend(pp_file)
                abnormal_labels.extend([[0]] * len(pp_file))
        
        
        normal_preprocessed = np.array(normal_preprocessed)
        abnormal_preprocessed = np.array(abnormal_preprocessed)
        
        normal_labels = np.array(normal_labels)
        abnormal_labels = np.array(abnormal_labels)
            
        self.preprocessed = np.concatenate((normal_preprocessed, abnormal_preprocessed), axis=0)
        self.preprocessed = np.expand_dims(self.preprocessed, axis=1)
        print(self.preprocessed.shape)
        
        self.labels = np.concatenate((normal_labels, abnormal_labels))
        print(self.labels.shape)
        

        self.num_files = len(normal_files) + len(abnormal_files)

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        
        return self.preprocessed[idx], self.labels[idx]
    
    
    
    
