# src/predict.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import pandas as pd
import random
from tqdm import tqdm
import logging
from .model import BiTGLSTMConfig, CNNConfig, AttentionConfig, BiTGLSTMModel, CNNModel, AttentionModel
from torch.utils.data import DataLoader
import os
import numpy as np

logger = logging.getLogger(__name__)

def load_model(model_path, config, model_type='lstm'):
    logger.info(f"Loading {model_type.upper()} model from {model_path}")

    if model_type == 'lstm':
        model = BiTGLSTMModel.from_pretrained(model_path, config=config)
    elif model_type == 'cnn':
        model = CNNModel.from_pretrained(model_path, config=config)
    elif model_type == 'attention':
        model = AttentionModel.from_pretrained(model_path, config=config)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention'.")

    model.eval()
    logger.info(f"{model_type.upper()} model loaded successfully")
    return model

def make_predictions(model, features, batch_size=32):
    """
    Generates predictions using the trained model on the provided features.

    Args:
        model (PreTrainedModel): The trained model.
        features (torch.Tensor): The input features.
        batch_size (int, optional): Batch size for prediction. Defaults to 32.

    Returns:
        np.ndarray: Array of predictions with shape (num_samples, max_len).
    """
    logger.info("Starting prediction process")
    dataloader = DataLoader(features, batch_size=batch_size, shuffle=False)
    all_predictions = []

    device = next(model.parameters()).device
    model.to(device)
    logger.critical(f"Model is on device: {device}")

    for batch in tqdm(dataloader, desc="Predicting"):
        # Ensure batch is on the same device as the model
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
        logits = outputs['logits']
        probabilities = logits.cpu().numpy()  # Move to CPU for aggregation
        all_predictions.append(probabilities)
    
    predictions = np.vstack(all_predictions)
    logger.info(f"Predictions generated with shape: {predictions.shape}")
    return predictions

def save_predictions_to_csv(original_features, predictions, labels=None, save_path='predictions.csv'):
    """
    Saves the predictions and corresponding labels to a CSV file.
    If labels are not provided, only predictions are saved.

    Args:
        original_features (torch.Tensor): Array of original features with shape (num_samples, max_len, 2).
        predictions (np.ndarray): Array of predictions with shape (num_samples, max_len).
        labels (torch.Tensor or None, optional): Ground truth labels. Defaults to None.
        save_path (str, optional): Path to save the CSV file. Defaults to 'predictions.csv'.
    """
    logger.info(f"Saving predictions to {save_path}")

    # Convert labels tensor to numpy array if labels are provided
    if labels is not None:
        labels_np = labels.cpu().numpy()
    else:
        labels_np = None

    # Create a DataFrame with predictions and labels
    # Each row corresponds to a time step for a specific sample
    num_samples, max_len = predictions.shape
    data = {
        'Sample_Index': [],
        'Time_Step': [],
        'Displacements': [],
        'Prediction': []
    }

    if labels_np is not None:
        data['Ground_Truth'] = []

    for sample_idx in range(num_samples):
        for time_step in range(max_len):
            data['Sample_Index'].append(sample_idx)
            data['Time_Step'].append(time_step)
            data['Displacements'].append(float(original_features[sample_idx, time_step].item()))
            data['Prediction'].append(predictions[sample_idx, time_step])
            if labels_np is not None:
                data['Ground_Truth'].append(labels_np[sample_idx, time_step])

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    logger.info(f"Predictions successfully saved to {save_path}")

def plot_model_output(model, original_features, features, labels=None, sample_index=None, model_name="Model", save_path='model_output_sample.png'):
    """
    Plots the model output for a specific sample and saves it to a file.
    If labels are not provided, ground truth lines are not plotted.

    Args:
        model (PreTrainedModel): The trained model.
        original_features (torch.Tensor): The original feature before removing gaps.
        features (torch.Tensor): The input features.
        labels (torch.Tensor or None, optional): The ground truth labels. Defaults to None.
        sample_index (int, optional): Index of the sample to plot. Defaults to a random sample.
        model_name (str, optional): Name of the model for the plot title. Defaults to "Model".
        save_path (str, optional): Path to save the plot image. Defaults to 'model_output_sample.png'.
    """
    if sample_index is None:
        sample_index = random.randint(0, len(features) - 1)
    logger.info(f"Plotting model output for sample index {sample_index}")

    # Determine the device of the model
    device = next(model.parameters()).device
    logger.debug(f"Model is on device: {device}")

    input_tensor = features[sample_index].unsqueeze(0).to(device)  # Move to model's device
    with torch.no_grad():
        outputs = model(input_tensor)
    logits = outputs['logits'].squeeze().cpu().numpy()  # Move logits back to CPU for plotting
    probabilities = logits  # Already sigmoid applied in model

    if labels is not None:
        ground_truth = labels[sample_index].cpu().numpy()
    else:
        ground_truth = None

    fig, ax = plt.subplots(figsize=(15, 5))
    ax2 = ax.twinx()

    # Assuming original_features is a tensor with shape [num_samples, max_len, num_features]
    feature_plot = original_features[sample_index].cpu().numpy()

    ax.plot(range(len(feature_plot)), feature_plot, color='r', alpha=0.5, label='Displacements')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Displacement Values', color='r')
    ax.tick_params(axis='y', labelcolor='r')

    ax2.plot(probabilities, color='b', label='Predicted Probabilities')
    if ground_truth is not None:
        for i, g in enumerate(ground_truth):
            if g == 1:
                # Only add 'Ground Truth' label once to the legend
                ax2.axvline(i, color='g', linestyle='--', alpha=0.5, label='Ground Truth' if i == 0 else "")
    ax2.set_ylabel('Probability / Ground Truth', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title(f"{model_name}: Model Output for Sample {sample_index}")
    lines, labels_ = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Combine legends, avoiding duplicate 'Ground Truth' labels
    combined = list(zip(lines + lines2, labels_ + labels2))
    seen = set()
    unique = []
    for line, label in combined:
        if label not in seen:
            unique.append((line, label))
            seen.add(label)
    if unique:
        lines, labels_ = zip(*unique)
        ax.legend(lines, labels_, loc='upper left')
    plt.tight_layout()

    plt.savefig(save_path)
    logger.info(f"Plot saved to {save_path}")
    plt.close(fig)  # Close the figure to free memory
