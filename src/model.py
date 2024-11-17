# src/model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import logging
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

def trim_labels(labels):
    logger.debug("Trimming labels to a maximum of 1.0")
    return torch.clamp(labels, max=1.0)


# =========================================================================================================
# LSTM Model
# =========================================================================================================

class LSTMConfig(PretrainedConfig):
    model_type = "lstm"

    def __init__(self, hidden_size=32, num_layers=1, max_len=150, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout
        logger.debug(f"LSTMConfig initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}, dropout={dropout}")

class LSTMModel(PreTrainedModel):
    def __init__(self, config, verbose=False):
        super().__init__(config)
        self.model = _LSTM(
            input_size=2, 
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers, 
            max_len=config.max_len,
            dropout=config.dropout
        )
        if verbose:
            logger.info("LSTMModel initialized")

    def forward(self, input_ids, labels=None):
        x = self.model(input_ids)
        logits = x  # No sigmoid here

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': torch.sigmoid(logits)}

class _LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, max_len=150, dropout=0.5):
        super(_LSTM, self).__init__()
        self.max_len = max_len
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2 * max_len)
        self.fc = nn.Linear(hidden_size * 2 * max_len, max_len)
        logger.debug(f"_LSTM initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}")

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]

        # Create mask where zeros indicate padding positions
        non_zero_mask = (x != 0).any(dim=2)  # Shape: [batch_size, seq_len]

        # Compute lengths by summing the non-zero positions in each sequence
        lengths = non_zero_mask.sum(dim=1)  # Shape: [batch_size]
        lengths = lengths.clamp(min=1)  # Avoid zero-length sequences

        # Pack the padded sequence
        packed_input = rnn_utils.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process with LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = rnn_utils.pad_packed_sequence(
            packed_output, batch_first=True, total_length=self.max_len
        )

        # Apply mask to zero out padding positions in the output
        mask = non_zero_mask.unsqueeze(2).expand_as(output)
        # plt.plot(output[0,:,])
        # plt.savefig('test_features_before.png')
        # plt.close()
        output = output * mask
        # plt.plot(output[0,:,])
        # plt.savefig('test_features_after.png')
        # plt.close()
        # breakpoint()

        # # Pooling: Mean over the sequence length
        # output_pooled = torch.mean(output, dim=1)  # Shape: [batch_size, hidden_size * 2]

        # Apply dropout and batch normalization
        # output_pooled = self.dropout(output_pooled)
        # output_pooled = self.batch_norm(output_pooled)       
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.batch_norm(output)
        # Pass through the fully connected layer
        x = self.fc(output)  # Shape: [batch_size, max_len]

        return x
    
# =========================================================================================================
# CNN Model
# =========================================================================================================

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
        self.fc = nn.Linear(((seq_length // 2)) * 32, seq_length)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        logger.debug(f"CNN1D initialized with seq_length={seq_length}, num_features={num_features}")

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, seq_length)
        x = self.avg_pool(x)
        x = self.tanh(self.conv(x))
        x = self.bn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

# =========================================================================================================
# Attention Model
# =========================================================================================================

class AttentionConfig(PretrainedConfig):
    model_type = "attention"
    
    def __init__(self, input_size=2, attention_dim=2048, max_len=276, dropout=0.1, **kwargs):
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
    
    def forward(self, inputs, mask=None):
        """
        Forward pass for the attention mechanism.
        
        Args:
            inputs (Tensor): Input tensor of shape [batch_size, seq_len, input_size]
            mask (Tensor, optional): Mask tensor of shape [batch_size, seq_len]
        
        Returns:
            context (Tensor): Context vector of shape [batch_size, input_size]
        """
        # Compute attention scores
        scores = self.attention(inputs).squeeze(-1)  # [batch_size, seq_len]
        
        if mask is not None:
            # Apply a large negative value to padded positions
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]
        weights = weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Compute the weighted sum of inputs
        context = torch.sum(inputs * weights, dim=1)  # [batch_size, input_size]
        return context

class AttentionConfig(PretrainedConfig):
    model_type = "attention_with_bilstm"
    
    def __init__(
        self, 
        input_size=2, 
        lstm_hidden_size=128, 
        lstm_num_layers=1, 
        lstm_dropout=0.1, 
        attention_dim=2048, 
        max_len=276, 
        dropout=0.1, 
        **kwargs
    ):
        """
        Configuration for the Attention model with a bidirectional LSTM.
        
        Args:
            input_size (int): Number of input features per time step.
            lstm_hidden_size (int): Hidden size of the LSTM.
            lstm_num_layers (int): Number of LSTM layers.
            lstm_dropout (float): Dropout rate between LSTM layers.
            attention_dim (int): Dimension of the attention layer.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate after attention.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.dropout = dropout
        logger.debug(
            f"AttentionConfig initialized with input_size={input_size}, "
            f"lstm_hidden_size={lstm_hidden_size}, lstm_num_layers={lstm_num_layers}, "
            f"lstm_dropout={lstm_dropout}, attention_dim={attention_dim}, "
            f"max_len={max_len}, dropout={dropout}"
        )

class Attention(nn.Module):
    def __init__(self, input_size, attention_dim):
        """
        Attention mechanism that computes attention weights and context vector.
        
        Args:
            input_size (int): Number of input features per time step (from LSTM).
            attention_dim (int): Dimension of the attention layer.
        """
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, inputs, mask=None):
        """
        Forward pass for the attention mechanism.
        
        Args:
            inputs (Tensor): Input tensor of shape [batch_size, seq_len, input_size]
            mask (Tensor, optional): Mask tensor of shape [batch_size, seq_len]
        
        Returns:
            context (Tensor): Context vector of shape [batch_size, input_size]
        """
        # Compute attention scores
        scores = self.attention(inputs).squeeze(-1)  # [batch_size, seq_len]
        
        if mask is not None:
            # Apply a large negative value to padded positions
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]
        weights = weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Compute the weighted sum of inputs
        context = torch.sum(inputs * weights, dim=1)  # [batch_size, input_size]
        return context

class AttentionModel(PreTrainedModel):
    config_class = AttentionConfig

    def __init__(self, config, verbose=False):
        """
        Attention model with a bidirectional LSTM before the attention mechanism.
        
        Args:
            config (AttentionConfig): Configuration object.
            verbose (bool): If True, logs additional information.
        """
        super().__init__(config)
        # Initialize the bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        logger.debug(
            f"LSTM initialized with hidden_size={config.lstm_hidden_size}, "
            f"num_layers={config.lstm_num_layers}, dropout={config.lstm_dropout}, "
            f"bidirectional=True"
        )
        
        # Update the input size for attention based on bidirectional LSTM
        attention_input_size = config.lstm_hidden_size * 2  # because bidirectional
        self.attention = Attention(input_size=attention_input_size, attention_dim=config.attention_dim)
        logger.debug(
            f"Attention layer initialized with input_size={attention_input_size}, "
            f"attention_dim={config.attention_dim}"
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(attention_input_size, config.max_len)
        self.sigmoid = nn.Sigmoid()
        
        if verbose:
            logger.info("AttentionModel initialized with bidirectional LSTM")

    def _generate_attention_mask(self, input_ids, padding_value=0):
        """
        Generates an attention mask for the input tensor.
        
        Args:
            input_ids (Tensor): Input tensor of shape [batch_size, seq_len, input_size]
            padding_value (int, optional): The value used for padding. Defaults to 0.
        
        Returns:
            Tensor: Attention mask of shape [batch_size, seq_len]
        """
        # Create mask where positions are True if not all features are equal to padding_value
        mask = (input_ids.abs().sum(dim=-1) != padding_value).to(torch.bool)  # [batch_size, seq_len]
        return mask

    def forward(self, input_ids, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (Tensor): Input tensor of shape [batch_size, seq_len, input_size]
            labels (Tensor, optional): Labels for computing the loss.
        
        Returns:
            dict: Dictionary containing loss and logits.
        """
        # Generate attention mask internally
        attention_mask = self._generate_attention_mask(input_ids)  # [batch_size, seq_len]
        if torch.isnan(attention_mask).any():
            logger.warning("Attention mask contains NaN values.")

        # Pass inputs through the bidirectional LSTM
        lstm_output, _ = self.lstm(input_ids)  # lstm_output: [batch_size, seq_len, hidden_size * 2]
        if torch.isnan(lstm_output).any():
            logger.warning("LSTM output contains NaN values.")

        # Compute context vector using attention mechanism
        context = self.attention(lstm_output, mask=attention_mask)  # [batch_size, hidden_size * 2]
        if torch.isnan(context).any():
            logger.warning("Context vector contains NaN values.")

        # Apply dropout
        dropped = self.dropout(context)
        
        # Compute logits
        logits = self.fc(dropped)  # [batch_size, max_len]
        logits = self.sigmoid(logits)  # [batch_size, max_len]
        
        loss = None
        if labels is not None:
            # Ensure labels have the same shape as logits
            if labels.shape != logits.shape:
                raise ValueError(f"Labels shape {labels.shape} does not match logits shape {logits.shape}.")
            loss_fct = nn.BCELoss()
            try:
                loss = loss_fct(logits, labels)
            except Exception as e:
                raise ValueError(f"Loss computation failed: {e}")
        
        return {'loss': loss, 'logits': logits}