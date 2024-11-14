# src/model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import logging

logger = logging.getLogger(__name__)

def trim_labels(labels):
    logger.debug("Trimming labels to a maximum of 1.0")
    return torch.clamp(labels, max=1.0)

# Existing LSTM Models
class LSTMConfig(PretrainedConfig):
    model_type = "lstm"

    def __init__(self, hidden_size=64, num_layers=1, max_len=276, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        logger.debug(f"LSTMConfig initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}")

class LSTMModel(PreTrainedModel):
    def __init__(self, config, verbose=False):
        super().__init__(config)
        self.model = _LSTM(input_size=2, hidden_size=config.hidden_size, num_layers=config.num_layers, max_len=config.max_len)
        self.sigmoid = nn.Sigmoid()
        if verbose:
            logger.info("LSTMModel initialized")

    def forward(self, input_ids, labels=None):
        x = self.model(input_ids)
        logits = self.sigmoid(x)
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

class _LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, num_layers=1, max_len=276):
        super(_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 * max_len, max_len)
        logger.debug(f"_LSTM initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}")

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# Existing CNN Models
class CNNConfig(PretrainedConfig):
    model_type = "cnn"

    def __init__(self, seq_length=300, num_features=2, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.num_features = num_features
        logger.debug(f"CNNConfig initialized with seq_length={seq_length}, num_features={num_features}")

class CNNModel(PreTrainedModel):
    def __init__(self, config, verbose=False):
        super().__init__(config)
        self.model = CNN1D(config.seq_length, config.num_features)
        self.sigmoid = nn.Sigmoid()
        if verbose:
            logger.info("CNNModel initialized")

    def forward(self, input_ids, labels=None):
        x = self.model(input_ids)
        logits = self.sigmoid(x)
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

class CNN1D(nn.Module):
    def __init__(self, seq_length, num_features=2):
        super(CNN1D, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.avg_pool = nn.AvgPool1d(kernel_size=30, stride=1, padding=14)
        self.conv = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=(seq_length // 2) + 1)
        self.pool = nn.MaxPool1d(kernel_size=30, stride=1, padding=15)
        self.fc = nn.Linear(((seq_length // 2)) * 32, seq_length)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        logger.debug(f"CNN1D initialized with seq_length={seq_length}, num_features={num_features}")

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, seq_length)
        x = self.avg_pool(x)
        x = self.tanh(self.conv(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# ----- New Attention LSTM Models -----
class AttentionLSTMConfig(PretrainedConfig):
    model_type = "attention_lstm"

    def __init__(self, hidden_size=64, num_layers=1, max_len=276, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout
        logger.debug(f"AttentionLSTMConfig initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}, dropout={dropout}")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size*2]
        scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(lstm_output * weights, dim=1)  # [batch_size, hidden_size*2]
        return context

class AttentionLSTMModel(PreTrainedModel):
    def __init__(self, config, verbose=False):
        super().__init__(config)
        self.model = _AttentionLSTM(input_size=2, hidden_size=config.hidden_size, num_layers=config.num_layers, max_len=config.max_len, dropout=config.dropout)
        self.sigmoid = nn.Sigmoid()
        if verbose:
            logger.info("AttentionLSTMModel initialized")

    def forward(self, input_ids, labels=None):
        x = self.model(input_ids)
        logits = self.sigmoid(x)
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

class _AttentionLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, max_len=276, dropout=0.1):
        super(_AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, max_len)  # *2 for bidirectional
        logger.debug(f"_AttentionLSTM initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}, dropout={dropout}")

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size*2]
        context = self.attention(lstm_out)  # context: [batch_size, hidden_size*2]
        logits = self.fc(context)  # logits: [batch_size, max_len]
        return logits