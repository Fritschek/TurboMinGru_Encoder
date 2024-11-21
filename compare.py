import time  # Import time module for measuring execution time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def exists(val):
    return val is not None

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

class MinGRUHeinsen(Module):
    def __init__(self, input_size, hidden_size, expansion_factor=1.0):
        super().__init__()

        dim_inner = int(hidden_size * expansion_factor)
        self.to_hidden_and_gate = Linear(input_size, dim_inner * 2, bias=False)
        self.to_out = Linear(dim_inner, hidden_size, bias=False) if expansion_factor != 1.0 else Identity()
        self.hidden_size = hidden_size

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # Handle sequential inference
            hidden = g(hidden)
            gate = gate.sigmoid()
            if exists(prev_hidden):
                out = torch.lerp(prev_hidden, hidden, gate)
            else:
                out = hidden * gate
        else:
            # Parallel computation
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))
            else:
                # Initialize prev_hidden if not provided
                prev_hidden = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden

# Standard GRU Model
class StandardGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True):
        super(StandardGRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=False
        )

    def forward(self, x, h_0=None):
        output, h_n = self.gru(x, h_0)
        return output

# Profiling Function with Total Timing
def profile_model(model, x, model_name="Model"):
    model.to(device)
    model.train()  # Ensure the model is in training mode

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Warm-up iterations
    for _ in range(2):
        optimizer.zero_grad()
        output = model(x)
        logits = output[:, -1, :]  # Use the last time step
        # For demonstration, let's assume we have 10 classes
        fc = nn.Linear(logits.size(-1), 10).to(device)
        logits = fc(logits)
        labels = torch.randint(0, 10, (x.size(0),)).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # Start timing
    start_time = time.time()

    # Start profiling
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
        for _ in range(50):
            optimizer.zero_grad()
            output = model(x)
            logits = output[:, -1, :]
            logits = fc(logits)
            labels = torch.randint(0, 10, (x.size(0),)).to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print the total execution time
    print(f"\nTotal execution time for {model_name}: {total_time:.4f} seconds")

    # Print the profiling results
    print(f"\nProfiling results for {model_name}:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

# Sample Data Generation
def generate_sample_data(batch_size, seq_len, input_size):
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 10, (batch_size,))
    return x, y

# Define Model Parameters
input_size = 10
hidden_size = 100
seq_len = 1000
batch_size = 128

# Generate input data
x, y = generate_sample_data(batch_size, seq_len, input_size)
x = x.to(device)
y = y.to(device)

# Instantiate MinGRUHeinsen Model
min_gru_heinsen_model = MinGRUHeinsen(
    input_size=input_size,
    hidden_size=hidden_size,
    expansion_factor=1.0
)

# Instantiate Standard GRU Model
standard_gru_model = StandardGRUModel(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    bias=True,
    batch_first=True
)

# Run Profiling for MinGRUHeinsen Model
profile_model(min_gru_heinsen_model, x, model_name="MinGRUHeinsen Model")

# Run Profiling for Standard GRU Model
profile_model(standard_gru_model, x, model_name="Standard GRU Model")

