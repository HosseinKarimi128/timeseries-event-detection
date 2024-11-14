# main.py

import logging
from src.utils import setup_logging
from src.data_processing import (
    create_mega_df,
    add_delta_t_features,
    sample_and_scale,
    CustomDataset,
    CustomDatasetCNN
)
from src.model import trim_labels
from src.train import train_model
from src.predict import load_model, make_predictions, save_predictions_to_csv, plot_model_output
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
import torch
import os

def train_model_gradio(
    labels_paths,
    features_paths,
    sample_size,
    epochs,
    batch_size,
    learning_rate,
    output_dir,
    model_type
):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting training via Gradio interface")

    labels_paths = [Path(path) for path in labels_paths]
    features_paths = [Path(path) for path in features_paths]
    max_len = 267  # Adjust based on your data

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Data Processing for Training
    mega_features, mega_labels = create_mega_df(labels_paths, features_paths, max_len)
    final_features = add_delta_t_features(mega_features)
    sampled_features, sampled_labels = sample_and_scale(final_features, mega_labels, sample_size=sample_size)

    # Label Trimming
    trimmed_labels = trim_labels(sampled_labels)

    # Train-validation split
    train_features, val_features, train_labels, val_labels = train_test_split(
        sampled_features, trimmed_labels, test_size=0.2, random_state=42
    )

    # Create datasets based on model type
    if model_type == 'lstm' or model_type == 'attention_lstm':
        train_dataset = CustomDataset(train_features, train_labels)
        val_dataset = CustomDataset(val_features, val_labels)
    elif model_type == 'cnn':
        train_dataset = CustomDatasetCNN(train_features, train_labels)
        val_dataset = CustomDatasetCNN(val_features, val_labels)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention_lstm'.")

    # Training
    trainer, metrics = train_model(
        train_dataset,
        val_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_type=model_type
    )
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # Extract metrics from the trainer
    return metrics  # Return metrics to display in Gradio

def predict_model_gradio(
    model_path,
    labels_paths,
    features_paths,
    sample_size,
    batch_size,
    predictions_csv,
    plot_save_path,
    save_plots,
    num_plot_samples,
    model_type
):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting prediction via Gradio interface")

    labels_paths = [Path(path) for path in labels_paths]
    features_paths = [Path(path) for path in features_paths]
    max_len = 267  # Adjust based on your data

    # Create output directory if it doesn't exist
    output_dir = Path(predictions_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data Processing for Prediction
    mega_features, mega_labels = create_mega_df(labels_paths, features_paths, max_len)
    final_features = add_delta_t_features(mega_features)
    sampled_features, sampled_labels = sample_and_scale(final_features, mega_labels, sample_size=sample_size)

    # Label Trimming
    trimmed_labels = trim_labels(sampled_labels)

    # Load configuration based on model type
    from src.model import LSTMConfig, CNNConfig, AttentionLSTMConfig

    if model_type == 'lstm':
        config = LSTMConfig.from_pretrained(model_path)
    elif model_type == 'cnn':
        config = CNNConfig.from_pretrained(model_path)
    elif model_type == 'attention_lstm':
        config = AttentionLSTMConfig.from_pretrained(model_path)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention_lstm'.")

    # Load model
    model = load_model(model_path, config, model_type=model_type)

    # Prediction
    predictions = make_predictions(model, sampled_features, batch_size=batch_size)
    save_predictions_to_csv(predictions, trimmed_labels, save_path=predictions_csv)

    plots = []
    if save_plots:
        # Generate multiple plots
        num_plot = num_plot_samples
        total_samples = len(sampled_features)

        if num_plot > total_samples:
            logger.warning(f"Requested number of plots ({num_plot}) exceeds the number of available samples ({total_samples}). Reducing to {total_samples}.")
            num_plot = total_samples

        # Select unique random sample indices
        sample_indices = random.sample(range(total_samples), k=num_plot)

        for sample_idx in sample_indices:
            # Define a unique save path for each plot
            base_save_path = Path(plot_save_path)
            save_dir = base_save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create a save path with sample index
            save_path = save_dir / f"{base_save_path.stem}_sample_{sample_idx}{base_save_path.suffix}"

            # Generate plot for the sample
            plot_model_output(
                model=model,
                features=sampled_features,
                labels=trimmed_labels,
                sample_index=sample_idx,
                model_name=model_type.upper(),
                save_path=str(save_path)
            )
            logger.info(f"Plot saved to {save_path}")
            plots.append(str(save_path))  # Collect plot paths

    logger.info("Prediction process completed.")
    return predictions_csv, plots  # Return results to display in Gradio
