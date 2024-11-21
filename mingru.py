import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility functions
def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# Log-space compatible minGRU
class LogSpaceMinGRU(nn.Module):
    def __init__(self, dim, expansion_factor=1.):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False) if expansion_factor != 1. else nn.Identity()

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            prev_hidden: Initial hidden state of shape (batch_size, 1, dim)
            return_next_prev_hidden: Boolean to return the next hidden state

        Returns:
            out: Output tensor of shape (batch_size, seq_len, dim)
            next_prev_hidden: Final hidden state of shape (batch_size, 1, dim)
        """
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if self.training:  # Use parallel log-space operations during training
            # Parallel mode using log-space operations
            log_coeffs = -F.softplus(gate)  # log(1 - z)
            log_z = -F.softplus(-gate)  # log(z)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            # Include the initial hidden state in log-space if provided
            if prev_hidden is not None:
                log_values = torch.cat((log_g(prev_hidden), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            # Use Heinsen's associative scan in log-space
            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]  # Only take the final output sequence

        else:  # Use sequential operations during inference
            # Sequential mode for single step processing
            out = []
            h_t = prev_hidden if prev_hidden is not None else torch.zeros_like(x[:, 0, :])
            for t in range(seq_len):
                hidden_t = g(hidden[:, t, :])
                gate_t = torch.sigmoid(gate[:, t, :])
                h_t = (1 - gate_t) * h_t + gate_t * hidden_t
                out.append(h_t.unsqueeze(1))

            out = torch.cat(out, dim=1)  # Concatenate along sequence dimension

        next_prev_hidden = out[:, -1:]  # Save the last hidden state for the next step

        # Apply a linear transformation to get the final output
        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out
        return out, next_prev_hidden

# Define the StackedMinGRULM class using the log-space compatible LogSpaceMinGRU
class StackedMinGRULM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, ff_mult=4, min_gru_expansion=1.5, conv_kernel_size=3, enable_conv=False):
        super(StackedMinGRULM, self).__init__()

        # Initialize model parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create a stack of layers for forward and backward directions
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            layer_list = nn.ModuleList()  # Use nn.ModuleList instead of a standard list
            for direction in range(self.num_directions):
                # Each component within the layer (Conv, LayerNorm, minGRU, etc.) must be wrapped in an nn.ModuleList
                direction_layer = nn.ModuleList([
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=conv_kernel_size, groups=hidden_size, padding=conv_kernel_size - 1) if enable_conv else None,
                    nn.LayerNorm(hidden_size),
                    LogSpaceMinGRU(input_size if _ == 0 else hidden_size, expansion_factor=min_gru_expansion),
                    nn.LayerNorm(hidden_size),
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size * ff_mult),
                        nn.GELU(),
                        nn.Linear(hidden_size * ff_mult, hidden_size)
                    )
                ])
                layer_list.append(direction_layer)  # Append the direction_layer (which is an nn.ModuleList) to layer_list
            self.layers.append(layer_list)  # Append the entire layer_list (which is an nn.ModuleList) to self.layers

        self.norm = nn.LayerNorm(hidden_size)  # Final normalization layer
        self.to_logits = nn.Linear(hidden_size * self.num_directions, 2, bias=False)  # Output layer for binary classification (logits)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            output: Tensor of shape (batch_size, seq_len, hidden_size * num_directions)
        """
        batch_size, seq_len, _ = input.size()

        # Initialize hidden states if not provided
        h_0 = torch.zeros(self.num_layers, self.num_directions, batch_size, self.hidden_size, device=input.device)

        next_prev_hiddens = []
        layer_output = input  # Start with input as layer input

        # Apply each layer sequentially
        for layer in self.layers:
            outputs = []

            # Process forward and backward directions separately
            for direction_idx, (conv, norm1, mingru, norm2, ff) in enumerate(layer):
                x = layer_output

                # Apply convolution if enabled
                if conv is not None:
                    x = x.transpose(1, 2)  # Convert to (batch_size, hidden_size, seq_len) for Conv1D
                    x = conv(x) + x  # Residual connection after convolution
                    x = x.transpose(1, 2)  # Convert back to (batch_size, seq_len, hidden_size)

                # Apply LogSpaceMinGRU with normalization
                h_i = h_0[:, direction_idx, :, :]
                min_gru_out, _ = mingru(norm1(x), h_i, return_next_prev_hidden=True)
                x = min_gru_out + x  # Residual connection after LogSpaceMinGRU

                # Apply feedforward network with normalization
                x = ff(norm2(x)) + x  # Residual connection after feedforward

                outputs.append(x)

            # Concatenate outputs from forward and backward directions if bidirectional
            if self.bidirectional:
                layer_output = torch.cat(outputs, dim=-1)  # Shape: (batch_size, seq_len, hidden_size * num_directions)
            else:
                layer_output = outputs[0]  # Only forward direction output

        # Final layer normalization
        embed = self.norm(layer_output)
        
        # Project the output to binary logits
        logits = self.to_logits(embed)  # Shape: (batch_size, seq_len, num_tokens)

        return logits
