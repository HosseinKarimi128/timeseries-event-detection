# src/data_processing.py

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
tqdm.pandas()
logger = logging.getLogger(__name__)


def cumulate_ones(_series):
    _series = pd.Series(_series)
    _output = _series.groupby((_series == 0).cumsum()).cumsum() * _series
    return _output

def create_delta_t(_series):
    _series = pd.Series(_series)
    _output = _series.copy()
    _output.iloc[:-1] = _output.iloc[:-1].values - _output.iloc[1:].values
    _output.iloc[:-1] = _output.iloc[:-1].clip(lower=0)
    _output.iloc[-1] = 0
    return _output

def cumulate_ones_and_create_delta_t(_series):
    return create_delta_t(cumulate_ones(_series))

def create_delta_t_df(_paths):
    for _path in _paths:
        file_name = str(_path).split('/')[-1]
        delta_t_file_path = os.path.join('data', 'delta_t_' + file_name)
        if os.path.exists(delta_t_file_path):
            logger.info(f"{delta_t_file_path} already exists")
            continue
        _df = pd.read_csv(_path, header=None)
        mask = _df.isna().astype(int)
        delta_t_df = mask.progress_apply(cumulate_ones_and_create_delta_t, axis=1)
        delta_t_df.to_csv(delta_t_file_path, index=False, header=False)
        logger.info(f"Created delta_t_{_path}")

def create_features_tensor(_paths, max_len):
    _features = []
    for _path in _paths:
        file_name = str(_path).split('/')[-1]
        delta_t_file_path = os.path.join('data', 'delta_t_' + file_name)
        _time_series = pd.read_csv(_path, header=None)
        _delta_t_series = pd.read_csv(delta_t_file_path, header=None)
        # pad or truncate to max_len
        if len(_time_series.columns) > max_len:
            _time_series = _time_series.iloc[:, :max_len]
            _delta_t_series = _delta_t_series.iloc[:, :max_len]
            _original_time_series = _time_series.copy()
        else:
            # Fix: Use pad_width to ensure consistent array shapes
            pad_width = max_len - len(_time_series.columns)
            _time_series = np.pad(_time_series.values, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
            _original_time_series = _time_series.copy()
            _delta_t_series = np.pad(_delta_t_series.values, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        
        _features.append(_time_series)
        _features.append(_delta_t_series)
    
    # Stack all features into a single numpy array before converting to tensor
    _features = np.stack(_features, axis=-1)
    return torch.tensor(_features, dtype=torch.float32), torch.tensor(_original_time_series, dtype=torch.float32)

def create_labels_tensor(_paths, max_len):
    _labels = []
    for _path in _paths:
        _df = pd.read_csv(_path, header=None)
        # pad or truncate to max_len
        if len(_df.columns) > max_len:
            _df = _df.iloc[:, :max_len]
        else:
            # Use same padding approach as features tensor
            pad_width = max_len - len(_df.columns)
            _df = np.pad(_df.values, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        _labels.append(_df)
    
    # Stack labels into single numpy array before converting to tensor
    _labels = np.stack(_labels, axis=-1)
    _labels = _labels.reshape(len(_labels), -1)
    return torch.tensor(_labels, dtype=torch.float32)
# Custom Datasets

def calculate_non_zero_length(features):
    non_zero_lengths = []
    for i in range(features.size(0)):
        non_zero_length = torch.sum(features[i, :, 0] != 0)
        non_zero_lengths.append(non_zero_length)
    return torch.tensor(non_zero_lengths)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.lengths = calculate_non_zero_length(features)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'input_ids': self.features[idx], 'lengths': self.lengths[idx], 'labels': self.labels[idx]}

class CustomDatasetCNN(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'input_ids': self.features[idx], 'labels': self.labels[idx]}
