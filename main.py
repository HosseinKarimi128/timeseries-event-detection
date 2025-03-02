# main.py

import logging
from src.compute_metrics import compute_metrics
from src.utils import setup_logging
from src.data_processing import (
    create_delta_t_df,
    create_features_tensor,
    create_labels_tensor,
    CustomDataset,
    CustomDatasetCNN,
)
from src.model import trim_labels
from src.train import train_model
from src.predict import load_model, make_predictions, save_predictions_to_csv, plot_model_output
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
import torch
import os
from matplotlib import pyplot as plt

def train_model_gradio(
    labels_paths=['data/Gaussian_Cp_EGMS_L3_E27N51_100km_E_2018_2022_1.csv'],
    features_paths=['data/time_series_EGMS_L3_E27N51_100km_E_2018_2022_1.csv'],
    sample_size=10000,
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    output_dir='results',
    model_type='lstm'
):
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting training via Gradio interface")

    labels_paths = [Path(path) for path in labels_paths]
    features_paths = [Path(path) for path in features_paths]
    max_len = 300  # Adjust based on your data

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Data Processing for Training
    logger.info("Creating delta_t_df")
    create_delta_t_df(features_paths)
    logger.info("Creating features_tensor")
    features_tensor, _ = create_features_tensor(features_paths, max_len)
    logger.info("Creating labels_tensor")
    labels_tensor = create_labels_tensor(labels_paths, max_len)
    print(features_tensor.shape)
    print(labels_tensor.shape)
    trimmed_labels = trim_labels(labels_tensor)
    # nan to 0
    features_tensor = torch.nan_to_num(features_tensor, nan=0.0)
    labels_tensor = torch.nan_to_num(labels_tensor, nan=0.0)
    #sample dataset
    sample_indices = random.sample(range(len(features_tensor)), k=sample_size)
    sampled_features = features_tensor[sample_indices]
    sampled_labels = trimmed_labels[sample_indices]
    # Train-validation split
    train_features, val_features, train_labels, val_labels = train_test_split(
        sampled_features, sampled_labels, test_size=0.2, random_state=42
    )

    # Create datasets based on model type
    if model_type == 'lstm' or model_type == 'attention':
        train_dataset = CustomDataset(train_features, train_labels)
        val_dataset = CustomDataset(val_features, val_labels)
    elif model_type == 'cnn':
        train_dataset = CustomDatasetCNN(train_features, train_labels)
        val_dataset = CustomDatasetCNN(val_features, val_labels)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention'.")

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
    predictions = make_predictions(trainer.model, val_dataset[:]['input_ids'])
    # Extract metrics from the trainer
    evaluation_results = compute_metrics(predictions, val_dataset[:]['labels'])
    return metrics, evaluation_results  # Return metrics to display in Gradio

def predict_model_gradio(
    model_path,
    labels_paths,  # Can be an empty list if labels are not uploaded
    features_paths,
    sample_size,
    batch_size,
    predictions_csv,
    plot_save_path,
    save_plots,
    num_plot_samples,
    model_type,
    input_indices=None,
    ):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting prediction via Gradio interface")

    labels_paths = [Path(path) for path in labels_paths] if labels_paths else []
    features_paths = [Path(path) for path in features_paths]
    max_len = 300  # Adjust based on your data

    # Create output directory if it doesn't exist
    output_dir = Path(predictions_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Creating delta_t_df")
    create_delta_t_df(features_paths)
    logger.info("Creating features_tensor")
    features_tensor, original_features_tensor = create_features_tensor(features_paths, max_len)
    logger.info("Creating labels_tensor")
    labels_tensor = None
    if labels_paths:
        labels_tensor = create_labels_tensor(labels_paths, max_len)
        trimmed_labels = trim_labels(labels_tensor)
    else:
        trimmed_labels = None
    # nan to 0
    features_tensor = torch.nan_to_num(features_tensor, nan=0.0)
    if labels_tensor is not None:
        labels_tensor = torch.nan_to_num(labels_tensor, nan=0.0)
    #sample dataset
    if not input_indices:
        if sample_size is None:
            sample_size = len(features_tensor)
        sample_indices = random.sample(range(len(features_tensor)), k=sample_size)
        sampled_features = features_tensor[sample_indices]
        sampled_original_features = original_features_tensor[sample_indices]
    else:
        sampled_features = features_tensor[input_indices[0]:input_indices[1]]
        sampled_original_features = original_features_tensor[input_indices[0]:input_indices[1]]
    if labels_tensor is not None:
        if not input_indices:
            sampled_labels = trimmed_labels[sample_indices]
        else:
            sampled_labels = trimmed_labels[input_indices[0]:input_indices[1]]
    else:
        sampled_labels = None
    
    # zero to nan
    # sampled_original_features = original_features_tensor.clone()
    # sampled_original_features[sampled_original_features == 0.0] = float('nan')
    # Load configuration based on model type
    from src.model import BiTGLSTMConfig, CNNConfig, AttentionConfig

    if model_type == 'lstm':
        config = BiTGLSTMConfig.from_pretrained(model_path)
    elif model_type == 'cnn':
        config = CNNConfig.from_pretrained(model_path)
    elif model_type == 'attention':
        config = AttentionConfig.from_pretrained(model_path)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention'.")

    # Load model
    model = load_model(model_path, config, model_type=model_type)

    # Prediction
    predictions = make_predictions(model, sampled_features, batch_size=batch_size)
    save_predictions_to_csv(sampled_original_features, predictions, trimmed_labels, save_path=predictions_csv)

    plots = []
    if save_plots:
        # Generate multiple plots
        num_plot = num_plot_samples if num_plot_samples is not None else len(sampled_features)
        total_samples = len(sampled_features)

        if num_plot > total_samples:
            logger.warning(f"Requested number of plots ({num_plot}) exceeds the number of available samples ({total_samples}). Reducing to {total_samples}.")
            num_plot = total_samples

        # if input_indices is not None:
        #     sample_indices = list(range(input_indices[0], input_indices[1]))
        # else:
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
                original_features=sampled_original_features,
                features=sampled_features,
                labels=sampled_labels,  # Can be None
                sample_index=sample_idx,
                model_name=model_type.upper(),
                save_path=str(save_path)
            )
            logger.info(f"Plot saved to {save_path}")
            plots.append(str(save_path))  # Collect plot paths

    logger.info("Prediction process completed.")
    return predictions_csv, plots  # Return results to display in Gradio
