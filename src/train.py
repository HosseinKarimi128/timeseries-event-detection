# src/train.py

import logging
from transformers import Trainer, TrainingArguments
import torch
from .model import BiTGLSTMConfig, BiTGLSTMModel, CNNConfig, CNNModel, AttentionConfig, AttentionModel
from .data_processing import CustomDataset, CustomDatasetCNN
from torchinfo import summary

import os

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

logger = logging.getLogger(__name__)

def train_model(train_dataset, val_dataset, output_dir="results", epochs=100, batch_size=32, learning_rate=0.001, model_type='lstm'):
    if model_type == 'lstm':
        sample_input = (train_dataset[0:32]['input_ids'], train_dataset[0:32]['lengths'])
        config = BiTGLSTMConfig(hidden_size=train_dataset.features.shape[1], num_layers=1, max_len=train_dataset.features.shape[1])
        model = BiTGLSTMModel(config=config, verbose=True)
    elif model_type == 'cnn':
        sample_input = (train_dataset[0:32]['input_ids'])
        config = CNNConfig(seq_length=train_dataset.features.shape[1], num_features=train_dataset.features.shape[2])
        model = CNNModel(config=config, verbose=True)
    elif model_type == 'attention':
        sample_input = (train_dataset[0:32]['input_ids'])
        config = AttentionConfig(
                    input_size=2,
                    lstm_hidden_size=256,
                    lstm_num_layers=1,
                    lstm_dropout=0.5,
                    attention_dim=512,
                    bidirectional=True,
                    output_dim=1,
                    dropout=0.5
                )
        model = AttentionModel(config=config)
    else:
        raise ValueError("Unsupported model type. Choose 'lstm', 'cnn', or 'attention'.")
    

    print(summary(model, input_data = (sample_input)))
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=1000,
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