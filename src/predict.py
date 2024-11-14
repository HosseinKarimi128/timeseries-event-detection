# src/predict.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import pandas as pd
import random
from tqdm import tqdm
import logging
from .model import LSTMConfig, LSTMModel, CNNConfig, CNNModel, AttentionLSTMConfig, AttentionLSTMModel
from torch.utils.data import DataLoader
import os
import numpy as np

logger = logging.getLogger(__name__)

def load_model(model_path, config, model_type='lstm'):
    logger.info(f"Loading {model_type.upper()} model from {model_path}")

    if model_type == 'lstm':
        model = LSTMModel.from_pretrained(model_path, config=config)
    elif model_type == 'cnn':
        model = CNNModel.from_pretrained(model_path, config=config)
    elif model_type == 'attention_lstm':
        model = AttentionLSTMModel.from_pretrained(model_path, config=config)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention_lstm'.")

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

    for batch in tqdm(dataloader, desc="Predicting"):
        # Ensure batch is on the same device as the model
        batch = batch.to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(batch)
        logits = outputs['logits']
        probabilities = logits.cpu().numpy()
        all_predictions.append(probabilities)
    
    predictions = np.vstack(all_predictions)
    logger.info(f"Predictions generated with shape: {predictions.shape}")
    return predictions

def save_predictions_to_csv(predictions, labels, save_path='predictions.csv'):
    """
    Saves the predictions and corresponding labels to a CSV file.
    
    Args:
        predictions (np.ndarray): Array of predictions with shape (num_samples, max_len).
        labels (torch.Tensor): Tensor of ground truth labels with shape (num_samples, max_len).
        save_path (str, optional): Path to save the CSV file. Defaults to 'predictions.csv'.
    """
    logger.info(f"Saving predictions to {save_path}")
    
    # Convert labels tensor to numpy array
    labels_np = labels.cpu().numpy()
    
    # Create a DataFrame with predictions and labels
    # Each row corresponds to a time step for a specific sample
    num_samples, max_len = predictions.shape
    data = {
        'Sample_Index': [],
        'Time_Step': [],
        'Prediction': [],
        'Ground_Truth': []
    }
    
    for sample_idx in range(num_samples):
        for time_step in range(max_len):
            data['Sample_Index'].append(sample_idx)
            data['Time_Step'].append(time_step)
            data['Prediction'].append(predictions[sample_idx, time_step])
            data['Ground_Truth'].append(labels_np[sample_idx, time_step])
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    logger.info(f"Predictions successfully saved to {save_path}")

def convert_zeros_to_nans(tensor):
    return tensor.masked_fill(tensor == 0, float('nan'))

def plot_model_output(model, features, labels, sample_index=None, model_name="Model", save_path='model_output_sample.png'):
    """
    Plots the model output for a specific sample and saves it to a file.

    Args:
        model (PreTrainedModel): The trained model.
        features (torch.Tensor): The input features.
        labels (torch.Tensor): The ground truth labels.
        sample_index (int, optional): Index of the sample to plot. Defaults to a random sample.
        model_name (str, optional): Name of the model for the plot title. Defaults to "Model".
        save_path (str, optional): Path to save the plot image. Defaults to 'model_output_sample.png'.
    """
    if sample_index is None:
        sample_index = random.randint(0, len(features) - 1)
    logger.info(f"Plotting model output for sample index {sample_index}")

    input_tensor = features[sample_index].unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    logits = outputs['logits'].squeeze().cpu().numpy()
    probabilities = torch.tensor(logits).numpy()
    ground_truth = labels[sample_index].cpu().numpy()

    fig, ax = plt.subplots(figsize=(25, 5))
    ax2 = ax.twinx()

    feature_plot = convert_zeros_to_nans(features[sample_index]).cpu().numpy()
    
    ax.scatter(range(len(feature_plot)), feature_plot[:, 0], color='r', alpha=0.5, label='Displacements')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Displacement Values', color='r')
    ax.tick_params(axis='y', labelcolor='r')

    ax2.plot(probabilities, color='b', label='Probabilities')
    for i,g in enumerate(ground_truth):
        if g == 1:
            ax2.axvline(i, color='g', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Probability / Ground Truth', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title(f"{model_name}: Model Output")
    lines, labels_ = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels_ + labels2, loc='upper left')
    plt.tight_layout()

    plt.savefig(save_path)
    logger.info(f"Plot saved to {save_path}")
    plt.close(fig)  # Close the figure to free memory
