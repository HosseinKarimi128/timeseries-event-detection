# main.py

import argparse
import logging
from src.utils import setup_logging
from src.data_processing import (
    create_mega_df,
    add_delta_t_features,
    sample_and_scale,
    remove_nan_from_features,
    CustomDataset,
    CustomDatasetCNN
)
from src.model import trim_labels
from src.train import train_model
from src.predict import load_model, make_predictions, save_predictions_to_csv, plot_model_output
from sklearn.model_selection import train_test_split
import random
from pathlib import Path  # Import Path for handling file paths

def main(args):
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Project started")

    # Data paths
    labels_paths = [
        'data/Gaussian_Cp_EGMS_L3_E27N51_100km_E_2018_2022_1.csv',
        'data/Gaussian_Cp_EGMS_L3_E27N51_100km_U_2018_2022_1.csv',
    ]

    features_paths = [
        'data/time_series_EGMS_L3_E27N51_100km_E_2018_2022_1.csv',
        'data/time_series_EGMS_L3_E27N51_100km_U_2018_2022_1.csv',
    ]

    max_len = 267  # Set to 267 to match the trained model

    # Data Processing
    mega_features, mega_labels = create_mega_df(labels_paths, features_paths, max_len)
    final_features = add_delta_t_features(mega_features)
    sampled_features, sampled_labels = sample_and_scale(final_features, mega_labels, sample_size=args.sample_size)

    # Label Trimming
    trimmed_labels = trim_labels(sampled_labels)

    # Train-validation split
    train_features, val_features, train_labels, val_labels = train_test_split(
        sampled_features, trimmed_labels, test_size=0.2, random_state=42
    )

    # Create datasets based on model type
    if args.model_type == 'lstm':
        train_dataset = CustomDataset(train_features, train_labels)
        val_dataset = CustomDataset(val_features, val_labels)
    elif args.model_type == 'cnn':
        train_dataset = CustomDatasetCNN(train_features, train_labels)
        val_dataset = CustomDatasetCNN(val_features, val_labels)
    elif args.model_type == 'attention_lstm':
        train_dataset = CustomDataset(train_features, train_labels)
        val_dataset = CustomDataset(val_features, val_labels)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention_lstm'.")

    if args.train:
        # Training
        trainer = train_model(
            train_dataset,
            val_dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_type=args.model_type
        )
        trainer.save_model(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")

    if args.predict:
        # Load configuration based on model type
        from src.model import LSTMConfig, CNNConfig, AttentionLSTMConfig

        if args.model_type == 'lstm':
            config = LSTMConfig.from_pretrained(args.model_path)
        elif args.model_type == 'cnn':
            config = CNNConfig.from_pretrained(args.model_path)
        elif args.model_type == 'attention_lstm':
            config = AttentionLSTMConfig.from_pretrained(args.model_path)
        else:
            raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention_lstm'.")

        # Prediction
        model = load_model(args.model_path, config, model_type=args.model_type)
        predictions = make_predictions(model, sampled_features, batch_size=args.batch_size)
        save_predictions_to_csv(predictions, trimmed_labels, save_path=args.predictions_csv)
        
        # Batch Sample Prediction: Generate multiple plots
        num_plot = args.num_plot_samples
        total_samples = len(sampled_features)
        
        if num_plot > total_samples:
            logger.warning(f"Requested number of plots ({num_plot}) exceeds the number of available samples ({total_samples}). Reducing to {total_samples}.")
            num_plot = total_samples
        
        # Select unique random sample indices
        sample_indices = random.sample(range(total_samples), k=num_plot)
        
        for idx, sample_idx in enumerate(sample_indices, start=1):
            # Define a unique save path for each plot
            base_save_path = Path(args.plot_save_path)
            save_dir = base_save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            
            # Create a save path with sample index
            save_path = save_dir / f"{base_save_path.stem}_sample_{sample_idx}{base_save_path.suffix}"
            
            # Generate plot for the sample
            plot_model_output(
                model=model,
                features=sampled_features,
                labels=trimmed_labels,
                sample_index=sample_idx,
                model_name=args.model_type.upper(),
                save_path=str(save_path)
            )
            logger.info(f"Plot saved to {save_path}")
        
        logger.info("Prediction process completed.")

    logger.info("Project completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPD Project with LSTM, CNN, and Attention LSTM Models")

    # Training arguments
    parser.add_argument('--train', action='store_true', help='Flag to trigger training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and prediction')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the trained model')

    # Prediction arguments
    parser.add_argument('--predict', action='store_true', help='Flag to trigger prediction')
    parser.add_argument('--model_path', type=str, default='results/model', help='Path to the trained model')
    parser.add_argument('--predictions_csv', type=str, default='results/predictions.csv', help='Path to save the predictions CSV')
    parser.add_argument('--plot_save_path', type=str, default='results/model_output_sample.png', help='Base path to save the prediction plots')

    # New argument for number of sample plots
    parser.add_argument('--num_plot_samples', type=int, default=1, help='Number of sample plots to generate during prediction')

    # General arguments
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to process for prediction')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'cnn', 'attention_lstm'], help='Type of model to use: "lstm", "cnn", or "attention_lstm"')

    args = parser.parse_args()
    main(args)

