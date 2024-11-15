# src/model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import logging
import torch.nn.utils.rnn as rnn_utils

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

# class _LSTM(nn.Module):
#     def __init__(self, input_size=2, hidden_size=8, num_layers=1, max_len=276):
#         super(_LSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True,
#                             bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2 * max_len, max_len)
#         logger.debug(f"_LSTM initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}")

#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#         return x
class _LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, num_layers=1, max_len=276):
        super(_LSTM, self).__init__()
        self.max_len = max_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 * max_len, max_len)
        logger.debug(f"_LSTM initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}")

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]

        # Create mask where zeros indicate padding positions
        # Assuming zeros are used only for padding and not actual data
        non_zero_mask = (x != 0).any(dim=2)  # Shape: [batch_size, seq_len]

        # Compute lengths by summing the non-zero positions in each sequence
        lengths = non_zero_mask.sum(dim=1)  # Shape: [batch_size]
        # Ensure lengths are at least 1 to avoid zero-length sequences
        lengths = lengths.clamp(min=1)
        # Pack the padded sequence
        packed_input = rnn_utils.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Process with LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = rnn_utils.pad_packed_sequence(
            packed_output, batch_first=True, total_length=self.max_len)
        
        # Apply mask to zero out padding positions in the output
        mask = non_zero_mask.unsqueeze(2).expand_as(output)
        output = output * mask

        # Flatten the output
        output_flat = output.contiguous().view(output.size(0), -1)

        # Pass through the fully connected layer
        x = self.fc(output_flat)

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
# class AttentionLSTMConfig(PretrainedConfig):
#     model_type = "attention_lstm"

#     def __init__(self, hidden_size=64, num_layers=1, max_len=276, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.max_len = max_len
#         self.dropout = dropout
#         logger.debug(f"AttentionLSTMConfig initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}, dropout={dropout}")

# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.attention = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

#     def forward(self, lstm_output):
#         # lstm_output: [batch_size, seq_len, hidden_size*2]
#         scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
#         weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
#         context = torch.sum(lstm_output * weights, dim=1)  # [batch_size, hidden_size*2]
#         return context

# class AttentionLSTMModel(PreTrainedModel):
#     def __init__(self, config, verbose=False):
#         super().__init__(config)
#         self.model = _AttentionLSTM(input_size=2, hidden_size=config.hidden_size, num_layers=config.num_layers, max_len=config.max_len, dropout=config.dropout)
#         self.sigmoid = nn.Sigmoid()
#         if verbose:
#             logger.info("AttentionLSTMModel initialized")

#     def forward(self, input_ids, labels=None):
#         x = self.model(input_ids)
#         logits = self.sigmoid(x)
#         loss = None
#         if labels is not None:
#             loss_fct = nn.BCELoss()
#             loss = loss_fct(logits, labels)
#         return {'loss': loss, 'logits': logits}

# class _AttentionLSTM(nn.Module):
#     def __init__(self, input_size=2, hidden_size=64, num_layers=1, max_len=276, dropout=0.1):
#         super(_AttentionLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.max_len = max_len
#         self.lstm = nn.LSTM(input_size=input_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True,
#                             bidirectional=True,
#                             dropout=dropout if num_layers > 1 else 0)
#         self.attention = Attention(hidden_size)
#         self.fc = nn.Linear(hidden_size * 2, max_len)  # *2 for bidirectional
#         logger.debug(f"_AttentionLSTM initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}, dropout={dropout}")

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size*2]
#         context = self.attention(lstm_out)  # context: [batch_size, hidden_size*2]
#         logits = self.fc(context)  # logits: [batch_size, max_len]
#         return logits

class AttentionConfig(PretrainedConfig):
    model_type = "attention"
    
    def __init__(self, input_size=2, attention_dim=64, max_len=276, dropout=0.1, **kwargs):
        """
        Configuration for the Attention-only model.
        
        Args:
            input_size (int): Number of input features per time step.
            attention_dim (int): Dimension of the attention layer.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.dropout = dropout
        logger.debug(
            f"AttentionConfig initialized with input_size={input_size}, "
            f"attention_dim={attention_dim}, max_len={max_len}, dropout={dropout}"
        )

class Attention(nn.Module):
    def __init__(self, input_size, attention_dim):
        """
        Attention mechanism that computes attention weights and context vector.
        
        Args:
            input_size (int): Number of input features per time step.
            attention_dim (int): Dimension of the attention layer.
        """
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, inputs):
        """
        Forward pass for the attention mechanism.
        
        Args:
            inputs (Tensor): Input tensor of shape [batch_size, seq_len, input_size]
        
        Returns:
            context (Tensor): Context vector of shape [batch_size, input_size]
        """
        # Compute attention scores
        scores = self.attention(inputs)  # [batch_size, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Compute the weighted sum of inputs
        context = torch.sum(inputs * weights, dim=1)  # [batch_size, input_size]
        return context

class AttentionModel(PreTrainedModel):
    config_class = AttentionConfig

    def __init__(self, config, verbose=False):
        """
        Attention-only model without LSTM.
        
        Args:
            config (AttentionConfig): Configuration object.
            verbose (bool): If True, logs additional information.
        """
        super().__init__(config)
        self.attention = Attention(input_size=config.input_size, attention_dim=config.attention_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.input_size, config.max_len)
        self.sigmoid = nn.Sigmoid()
        
        if verbose:
            logger.info("AttentionModel initialized without LSTM")
    
    def forward(self, input_ids, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (Tensor): Input tensor of shape [batch_size, seq_len, input_size]
            labels (Tensor, optional): Labels for computing the loss.
        
        Returns:
            dict: Dictionary containing loss and logits.
        """
        context = self.attention(input_ids)  # [batch_size, input_size]
        dropped = self.dropout(context)
        logits = self.fc(dropped)  # [batch_size, max_len]
        logits = self.sigmoid(logits)  # [batch_size, max_len]
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}