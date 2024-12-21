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

def create_mega_df(labels, features, max_len, cache_dir="cached_data", device="cpu"):
    """
    Creates combined DataFrames for features and labels with caching support.
    
    Args:
        labels (list of Path): List of label file paths. Can be empty.
        features (list of Path): List of feature file paths.
        max_len (int): Maximum sequence length.
        cache_dir (str): Directory to store cached files.
        device (str): Device to use for processing ('cpu' or 'cuda')

    Returns:
        tuple: (mega_features, mega_labels)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file paths
    features_cache = os.path.join(cache_dir, f"mega_features_len{max_len}.csv")
    labels_cache = os.path.join(cache_dir, f"mega_labels_len{max_len}.csv")
    
    # Check if cached files exist
    if os.path.exists(features_cache) and (not labels or os.path.exists(labels_cache)):
        logger.info("Loading from cached files")
        import dask.dataframe as dd
        
        # Read features and convert to pandas DataFrame
        mega_features = dd.read_csv(features_cache, 
                                  blocksize='64MB')\
                         .compute(scheduler='threads')[:, :max_len]  # Returns pandas DataFrame
        
        # Read labels if they exist and convert to pandas DataFrame
        if labels and os.path.exists(labels_cache):
            mega_labels = dd.read_csv(labels_cache,
                                    blocksize='64MB')\
                           .compute(scheduler='threads')[:, :max_len]  # Returns pandas DataFrame
        else:
            mega_labels = None
            
        logger.info("Successfully converted Dask DataFrames to pandas DataFrames")
        return mega_features, mega_labels

    logger.info("Cache not found. Creating mega DataFrame from labels and features")
    all_features = []
    all_labels = []

    # Add progress bar for features processing
    import dask.dataframe as dd
    
    # Process features files using dask
    logger.info("Processing features files with dask")
    dask_features = []
    for i in tqdm(range(len(features)), desc="Processing features"):
        features_path = features[i]
        # Read with dask
        ddf = dd.read_csv(features_path, header=None, blocksize='64MB', assume_missing=True)
        # Convert to pandas
        features_df = ddf.compute(scheduler='threads', assume_missing=True)
        
        if len(features_df.columns) < max_len:
            features_df = pad_data_frame(features_df, max_len) 
            logger.info(f"Padded features file {i+1}/{len(features)}: Features shape {features_df.shape}")
        else:
            features_df = truncate_data_frame(features_df, max_len)
            logger.info(f"Truncated features file {i+1}/{len(features)}: Features shape {features_df.shape}")
            
        all_features.append(features_df)
        logger.info(f"Processed features file {i+1}/{len(features)}: Features shape {features_df.shape}")

        if labels:
            if i < len(labels):
                labels_path = labels[i]
                # Read labels with dask
                ddf_labels = dd.read_csv(labels_path, header=None, blocksize='64MB', assume_missing=True)
                # Convert to pandas
                labels_df = ddf_labels.compute(scheduler='threads', assume_missing=True)

                if len(labels_df.columns) < max_len:
                    labels_df = pad_data_frame(labels_df, max_len)
                else:
                    labels_df = truncate_data_frame(labels_df, max_len)

                all_labels.append(labels_df)
                logger.info(f"Processed labels file {i+1}/{len(labels)}: Labels shape {labels_df.shape}")
            else:
                logger.warning(f"No corresponding label file for features file {i+1}. Filling with zeros.")
                dummy_labels = pd.DataFrame(0, index=range(max_len), columns=[0])
                all_labels.append(dummy_labels)
        else:
            logger.info(f"No labels provided for features file {i+1}. Filling labels with zeros.")
            dummy_labels = pd.DataFrame(0, index=range(max_len), columns=[0])
            all_labels.append(dummy_labels)

    if all_features:
        logger.info("Concatenating features...")
        mega_features = pd.concat(all_features, axis=0).reset_index(drop=True)
        logger.info(f"Mega Features shape: {mega_features.shape}")
        
        logger.info("Saving features cache...")
        mega_features.to_csv(features_cache, index=False)
        logger.info(f"Cached features saved to {features_cache}")
    else:
        mega_features = pd.DataFrame()
        logger.warning("No features provided. Mega Features is empty.")

    if labels:
        logger.info("Concatenating labels...")
        mega_labels = pd.concat(all_labels, axis=0).reset_index(drop=True)
        logger.info(f"Mega Labels shape: {mega_labels.shape}")
        
        logger.info("Saving labels cache...")
        mega_labels.to_csv(labels_cache, index=False)
        logger.info(f"Cached labels saved to {labels_cache}")
    else:
        mega_labels = None
        logger.info("No labels provided. Mega Labels is set to None.")

    return mega_features, mega_labels

def create_delta_t_feature(sequence, max_len):
    sequence = np.nan_to_num(sequence)
    is_zero_mask = sequence != 0
    return count_ones_since_last_zero(is_zero_mask, max_len)

def count_ones_since_last_zero(arr, max_len):
    output = []
    count = 1
    for i in range(max_len):
        if arr[i]:
            output.append(count)
            count = 1
        else:
            output.append(1)
            count += 1
    return np.array(output)

def add_delta_t_features(mega_features):
    logger.info("Adding delta t features")
    max_len = mega_features.shape[1]
    delta_t = mega_features.progress_apply(lambda x: create_delta_t_feature(x, max_len), axis=1)
    final_features = np.dstack((mega_features.values, np.vstack(delta_t.values)))
    logger.info(f"Final features shape after adding delta t and length: {final_features.shape}")
    return final_features

def sample_and_scale(final_features, mega_labels, sample_size=1000):
    """
    Samples and scales the features and labels.

    Args:
        final_features (np.ndarray): Final features array with shape (num_samples, seq_len, num_features).
        mega_labels (pd.DataFrame or None): Mega labels DataFrame or None if labels are not provided.
        sample_size (int): Number of samples to select.

    Returns:
        tuple: (sampled_features, sampled_labels)
            sampled_features (torch.Tensor): Sampled and scaled features.
            sampled_labels (torch.Tensor or None): Sampled and scaled labels or None if labels are not provided.
    """
    # logger.info("Sampling and scaling features and labels")
    # num_available = len(final_features)
    # actual_sample_size = min(sample_size, num_available)
    # indices = np.random.choice(num_available, size=actual_sample_size, replace=False)
    # sampled_features = final_features[indices]

    # # Initialize sampled_labels as None
    # sampled_labels = None

    # if mega_labels is not None:
    #     sampled_labels = mega_labels.values[indices]

    # # # Separate features and delta_t features
    # # features = sampled_features[:, :, 0]
    # # delta_t = sampled_features[:, :, 1]

    # # # Scale features
    # # scaler_features = StandardScaler()
    # # features = scaler_features.fit_transform(features)

    # # # Combine scaled features with delta_t (assuming delta_t doesn't need scaling)
    # # scaled_features = np.stack((features, delta_t), axis=-1)

    sampled_features = torch.from_numpy(final_features[:sample_size]).float()

    if mega_labels is not None:
        sampled_labels = torch.from_numpy(mega_labels.values[:sample_size]).float()
        logger.info(f"Sampled Features shape: {sampled_features.shape}")
        logger.info(f"Sampled Labels shape: {sampled_labels.shape}")
    else:
        logger.info(f"Sampled Features shape: {sampled_features.shape}")
        logger.info("Sampled Labels: None")

    return sampled_features, sampled_labels

def just_scale(final_features, mega_labels):
    logger.info("Scaling features and labels")
    features = final_features[:, :, 0]
    delta_t = final_features[:, :, 1]
    not_nan_features = features[~np.isnan(features)]
    mean = np.mean(not_nan_features)
    std = np.std(not_nan_features)
    scaled_features = (features - mean) / (std)
    # Combine scaled features with delta_t (assuming delta_t doesn't need scaling)
    scaled_features = np.stack((scaled_features, delta_t), axis=-1)

    sampled_features = torch.from_numpy(scaled_features).float()

    if mega_labels is not None:
        sampled_labels = torch.from_numpy(mega_labels.values).float()
        logger.info(f"Sampled Features shape: {sampled_features.shape}")
        logger.info(f"Sampled Labels shape: {sampled_labels.shape}")
    else:
        logger.info(f"Sampled Features shape: {sampled_features.shape}")
        logger.info("Sampled Labels: None")

    return sampled_features, sampled_labels

def gap_removal(sampled_features, max_length=200):
    logger.info("Removing gaps from features")
    gap_mask = sampled_features[:, :, 0] != 0
    output_features = []

    for i in range(sampled_features.size(0)):
        mask = gap_mask[i]
        cleaned = sampled_features[i][mask]
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        else:
            padding = torch.zeros(max_length - len(cleaned), sampled_features.size(2))
            cleaned = torch.cat([cleaned, padding], dim=0)
        output_features.append(cleaned)

    output_features = torch.stack(output_features)
    logger.info(f"Output Features shape after gap removal: {output_features.shape}")
    return output_features

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
