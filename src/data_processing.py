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

def pad_data_frame(df, max_len):
    logger.debug(f"Padding DataFrame to max length {max_len}")
    return df.reindex(columns=range(max_len), fill_value=0)

def truncate_data_frame(df, max_len):
    logger.debug(f"Truncating DataFrame to max length {max_len}")
    return df.iloc[:, :max_len]

def create_mega_df(labels, features, max_len):
    logger.info("Creating mega DataFrame from labels and features")
    all_features = []
    all_labels = []

    for i in range(len(labels)):
        features_path = features[i]
        labels_path = labels[i]

        features_df = pd.read_csv(features_path, header=None)
        labels_df = pd.read_csv(labels_path, header=None)

        if len(features_df) < max_len:
            features_df = pad_data_frame(features_df, max_len)
        else:
            features_df = truncate_data_frame(features_df, max_len)

        if len(labels_df) < max_len:
            labels_df = pad_data_frame(labels_df, max_len)
        else:
            labels_df = truncate_data_frame(labels_df, max_len)

        all_features.append(features_df)
        all_labels.append(labels_df)
        logger.debug(f"Processed file {i+1}/{len(labels)}: Features shape {features_df.shape}")

    mega_features = pd.concat(all_features, axis=0).reset_index(drop=True)
    mega_labels = pd.concat(all_labels, axis=0).reset_index(drop=True)

    logger.info(f"Mega Features shape: {mega_features.shape}")
    logger.info(f"Mega Labels shape: {mega_labels.shape}")

    return mega_features, mega_labels

def create_delta_t_feature(sequence, max_len):
    sequence = np.nan_to_num(sequence)
    is_zero_mask = sequence == 0
    return count_ones_since_last_zero(is_zero_mask, max_len)

def count_ones_since_last_zero(arr, max_len):
    output = []
    count = 1
    for i in range(max_len):
        if arr[i]:
            output.append(count)
            count = 1
        else:
            output.append(0)
            count += 1
    return np.array(output)

def add_delta_t_features(mega_features):
    logger.info("Adding delta t features")
    max_len = mega_features.shape[1]
    delta_t = mega_features.progress_apply(lambda x: create_delta_t_feature(x, max_len), axis=1)
    final_features = np.dstack((mega_features.values, np.vstack(delta_t.values)))
    logger.info(f"Final features shape after adding delta t: {final_features.shape}")
    return final_features

def sample_and_scale(final_features, mega_labels, sample_size=1000):
    logger.info("Sampling and scaling features and labels")
    indices = np.random.choice(len(final_features), size=sample_size, replace=False)
    sampled_features = final_features[indices]
    sampled_labels = mega_labels.values[indices]

    # Separate features and delta_t features
    features = sampled_features[:, :, 0]
    delta_t = sampled_features[:, :, 1]

    # Scale features and delta_t separately
    scaler_features = StandardScaler()
    scaler_delta_t = StandardScaler()

    features = scaler_features.fit_transform(features)
    delta_t = scaler_delta_t.fit_transform(delta_t)

    # Combine scaled features
    scaled_features = np.stack((features, delta_t), axis=-1)

    sampled_features = torch.from_numpy(scaled_features).float()
    sampled_labels = torch.from_numpy(sampled_labels).float()

    logger.info(f"Sampled Features shape: {sampled_features.shape}")
    logger.info(f"Sampled Labels shape: {sampled_labels.shape}")

    return sampled_features, sampled_labels


def remove_nan_from_features(sampled_features, max_length=200):
    logger.info("Removing NaNs from features")
    non_nan_mask = ~torch.isnan(sampled_features[:, :, 0])
    output_features = []

    for i in range(sampled_features.size(0)):
        mask = non_nan_mask[i]
        cleaned = sampled_features[i][mask]
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        else:
            padding = torch.zeros(max_length - len(cleaned), sampled_features.size(2))
            cleaned = torch.cat([cleaned, padding], dim=0)
        output_features.append(cleaned)

    output_features = torch.stack(output_features)
    logger.info(f"Output Features shape after NaN removal: {output_features.shape}")
    return output_features

# Custom Datasets
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'input_ids': self.features[idx], 'labels': self.labels[idx]}

class CustomDatasetCNN(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'input_ids': self.features[idx], 'labels': self.labels[idx]}
