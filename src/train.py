# src/train.py

import logging
from transformers import Trainer, TrainingArguments
import torch
from .model import LSTMConfig, LSTMModel, CNNConfig, CNNModel, AttentionConfig, AttentionModel
from .data_processing import CustomDataset, CustomDatasetCNN
from torchinfo import summary

import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

logger = logging.getLogger(__name__)

def train_model(train_dataset, val_dataset, output_dir="results", epochs=100, batch_size=32, learning_rate=0.001, model_type='lstm'):
    if model_type == 'lstm':
        config = LSTMConfig(hidden_size=4, num_layers=1, max_len=train_dataset.features.shape[1])
        model = LSTMModel(config=config, verbose=True)
    elif model_type == 'cnn':
        config = CNNConfig(seq_length=train_dataset.features.shape[1], num_features=train_dataset.features.shape[2])
        model = CNNModel(config=config, verbose=True)
    elif model_type == 'attention':
        config = AttentionConfig(attention_dim=5000, num_layers=2, max_len=train_dataset.features.shape[1], dropout=0.3)
        model = AttentionModel(config=config, verbose=True)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention'.")
    sample_input = train_dataset[0:32]['input_ids']
    print(summary(model, input_data = (sample_input)))
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=10000,
        save_total_limit=2,
        logging_dir='logs/',
        logging_steps=100,
        report_to=["tensorboard"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    logger.info("Training completed")

    # Return trainer and metrics
    return trainer, trainer.state.log_history