# src/model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import logging
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

def trim_labels(labels):
    logger.debug("Trimming labels to a maximum of 1.0")
    return torch.clamp(labels, max=1.0)


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
        self.oddity = int(seq_length % 2 == 0)
        self.conv = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=(seq_length // 2) + self.oddity)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(((seq_length // 2) - self.oddity) * 32, seq_length)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        logger.debug(f"CNN1D initialized with seq_length={seq_length}, num_features={num_features}")

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, seq_length)
        x = self.avg_pool(x)
        x = self.tanh(self.conv(x))
        # x = self.max_pool(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class BiTGLSTMConfig(PretrainedConfig):
    model_type = "lstm"

    def __init__(self, hidden_size=32, num_layers=1, max_len=150, dropout=0.5, bidirectional=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout
        self.bidirectional = bidirectional
        logger.debug(f"LSTMConfig initialized with hidden_size={hidden_size}, num_layers={num_layers}, max_len={max_len}, dropout={dropout}, bidirectional={bidirectional}")

class BiTGLSTMModel(PreTrainedModel):
    def __init__(self, config, verbose=False):
        super().__init__(config)
        num_directions = 2 if config.bidirectional else 1
        self.model = TimeGatedLSTM(
            input_size=1, 
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers,
            bidirectional=config.bidirectional
        )
        self.output_layer = nn.Linear(config.hidden_size * num_directions, config.max_len)
        if verbose:
            logger.info("LSTMModel initialized")
        self.lstm = nn.LSTM(
            input_size=config.hidden_size*num_directions,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        self.config = config
    def forward(self, input_ids, lengths=None, labels=None):
        if lengths is None:
            lengths = torch.sum(input_ids != 0, dim=-2)[0]
        x = input_ids[:, :lengths[0], 0].unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
        delta_t = input_ids[:, :lengths[0], 1].unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
        x, hc = self.model(x, delta_t)
        # h, c = hc
        # h = torch.cat(h).view(32,self.config.num_layers*2,-1)[:,:,-1]
        # c = torch.cat(c).view(32,self.config.num_layers*2,-1)[:,:,-1]
        # breakpoint()
        x = torch.max(x, dim=1).values  # Shape: [batch_size, hidden_size * num_directions]
        x , _ = self.lstm(x)
        logits = self.output_layer(x).squeeze(-1)  # Shape: [batch_size]
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': torch.sigmoid(logits)}

class BiTimeGatedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiTimeGatedLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # Combined weight matrices and biases
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

        # Time-gating mechanism
        self.time_gate = nn.Linear(1, hidden_size)

    def forward(self, x, delta_t, hx):
        h_t, c_t = hx  # Initial hidden and cell states

        # Compute gates in a vectorized manner
        gates = self.weight_ih(x) + self.weight_hh(h_t).unsqueeze(1)

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=2)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        t_gate = torch.sigmoid(self.time_gate(delta_t))

        # Update cell state and hidden state
        c_t = f_gate * c_t.unsqueeze(1) + i_gate * g_gate * t_gate
        h_t = o_gate * torch.tanh(c_t)

        h_t_final = h_t[:, -1, :]  # Get the last time step
        c_t_final = c_t[:, -1, :]  # Get the last time step

        return h_t, (h_t_final, c_t_final)

class TimeGatedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(TimeGatedLSTM, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        num_directions = 2 if bidirectional else 1

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            for direction in range(num_directions):
                input_dim = input_size if layer_idx == 0 else hidden_size * num_directions
                self.layers.append(BiTimeGatedLSTMCell(input_dim, hidden_size))

    def forward(self, x, delta_t):
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        num_directions = 2 if self.bidirectional else 1

        if self.bidirectional:
            h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers * num_directions)]
            c = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers * num_directions)]

            output_forward = x
            output_backward = torch.flip(x, dims=[1])
            delta_t_backward = torch.flip(delta_t, dims=[1])

            for layer_idx in range(self.num_layers):
                layer_fwd_idx = layer_idx * num_directions
                layer_bwd_idx = layer_fwd_idx + 1

                # Forward pass
                layer_fwd = self.layers[layer_fwd_idx]
                output_forward, (h[layer_fwd_idx], c[layer_fwd_idx]) = layer_fwd(
                    output_forward, delta_t, (h[layer_fwd_idx], c[layer_fwd_idx])
                )

                # Backward pass
                layer_bwd = self.layers[layer_bwd_idx]
                output_backward, (h[layer_bwd_idx], c[layer_bwd_idx]) = layer_bwd(
                    output_backward, delta_t_backward, (h[layer_bwd_idx], c[layer_bwd_idx])
                )
                output_backward = torch.flip(output_backward, dims=[1])  # Reverse outputs back to original order

                # Concatenate outputs
                output = torch.cat((output_forward, output_backward), dim=2)

                # Prepare inputs for next layer
                output_forward = output
                output_backward = torch.flip(output, dims=[1])
                delta_t_backward = torch.flip(delta_t, dims=[1])

            return output, (h, c)
        else:
            h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]

            for layer_idx in range(self.num_layers):
                layer = self.layers[layer_idx]
                x, (h[layer_idx], c[layer_idx]) = layer(x, delta_t, (h[layer_idx], c[layer_idx]))

            return x


# =========================================================================================================
# Attention Model
# =========================================================================================================

class AttentionConfig(PretrainedConfig):
    model_type = "attention_with_bilstm"
    
    def __init__(
        self, 
        input_size=2, 
        lstm_hidden_size=128, 
        lstm_num_layers=1, 
        lstm_dropout=0.1, 
        attention_dim=128, 
        bidirectional=False,
        output_dim=1,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.attention_dim = attention_dim
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.dropout = dropout

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional=False, dropout=0.1):
        super(EncoderLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            bidirectional=bidirectional, 
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim * self.num_directions
    
    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        outputs, (hidden, cell) = self.rnn(src)
        # outputs: (batch_size, seq_len, hidden_dim)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(encoder_outputs))  # (batch_size, seq_len, attention_dim)
        attention = self.v(energy).squeeze(2)  # (batch_size, seq_len)
        return attention  # Return attention scores before softmax

class AttentionModel(PreTrainedModel):
    config_class = AttentionConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = EncoderLSTM(
            input_dim=config.input_size, 
            hidden_dim=config.lstm_hidden_size, 
            n_layers=config.lstm_num_layers, 
            bidirectional=config.bidirectional,
            dropout=config.lstm_dropout
        )
        self.hidden_dim = config.lstm_hidden_size * (2 if config.bidirectional else 1)
        self.attention = Attention(self.hidden_dim, config.attention_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.hidden_dim, config.output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, labels=None):
        # input_ids: (batch_size, seq_len, input_size)
        encoder_outputs, hidden, cell = self.encoder(input_ids)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        attention_scores = self.attention(encoder_outputs)  # (batch_size, seq_len)
        
        # Apply causal mask to prevent attention to future positions
        batch_size, seq_len, _ = encoder_outputs.size()
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(encoder_outputs.device)  # (1, seq_len, seq_len)
        attention_scores = attention_scores.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, seq_len)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Compute context vectors
        context_vectors = torch.bmm(attention_weights, encoder_outputs)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        context_vectors = self.dropout(context_vectors)
        
        # Predict probabilities for each time step
        predictions = self.fc(context_vectors).squeeze(-1)  # (batch_size, seq_len)
        predictions = self.sigmoid(predictions)
        
        loss = None
        if labels is not None:
            # Ensure labels have the same shape as predictions
            if labels.shape != predictions.shape:
                raise ValueError(f"Labels shape {labels.shape} does not match predictions shape {predictions.shape}.")
            loss_fct = nn.BCELoss()
            loss = loss_fct(predictions, labels)
        
        return {'loss': loss, 'logits': predictions}