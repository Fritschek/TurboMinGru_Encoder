import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
import logging
import numpy as np
from torch.nn import functional as F
#import matplotlib.pyplot as plt
#import torch_optimizer as optim    ----- for shampoo



# Data Generation Function
def generate_data(batch_size, sequence_length, num_symbols):
    return torch.randint(0, num_symbols, (batch_size, sequence_length), dtype=torch.float)

def compute_ber(decoded_output, inputs, num_symbols=None, mode="symbol"):
    """
    Calculate the Bit Error Rate (BER) for different types of decoded outputs.

    Arguments:
        decoded_output (torch.Tensor): Predicted outputs, either softmax/one-hot encoded or binary.
        inputs (torch.Tensor): Ground truth values, either symbol indices or binary values.
        num_symbols (int, optional): Number of unique symbols (required for "symbol" mode).
        mode (str): "symbol" for one-hot encoded outputs or "binary" for sigmoid outputs.

    Returns:
        float: Calculated BER.
    """
    if mode == "symbol":
        # Calculate Symbol Error Rate (SER) for softmax/one-hot encoded outputs
        if num_symbols is None:
            raise ValueError("num_symbols must be specified when mode is 'symbol'")

        # Get predicted symbols from softmax or one-hot encoded outputs
        predicted_symbols = torch.argmax(decoded_output, dim=-1)
        # Calculate the number of symbol errors
        symbol_errors = (predicted_symbols != inputs).sum().item()
        # Calculate Symbol Error Rate (SER)
        SER = symbol_errors / inputs.numel()
        
        # Convert SER to BER
        bits_per_symbol = np.log2(num_symbols)
        BER = SER * bits_per_symbol

    elif mode == "binary":
        # Calculate BER for binary (sigmoid) outputs
        # Convert sigmoid outputs to binary values
        binary_predictions = torch.round(decoded_output)
        # Calculate the bitwise errors
        prediction_errors = torch.ne(binary_predictions, inputs)
        # Compute the BER as the mean of bitwise errors
        BER = torch.mean(prediction_errors.float()).detach().cpu().item()
    else:
        raise ValueError("Unsupported mode. Choose 'symbol' or 'binary'.")
    
    return BER

# AWGN Channel Simulation Function
def awgn_channel(encoded_data, ebno_db, rate, device, decoder_training=False, ebno_range=(-3.5, 0)):
    if decoder_training:
        low_ebno_db = ebno_db + ebno_range[0]
        high_ebno_db = ebno_db + ebno_range[1]
        ebno_db_matrix = np.random.uniform(low_ebno_db, high_ebno_db, size=encoded_data.shape)
        ebno_linear = 10**(ebno_db_matrix / 10)
        ebno_linear = torch.from_numpy(ebno_linear).float().to(device)
    else:
        ebno_linear = 10**(ebno_db / 10)
        
    signal_power = torch.mean(encoded_data**2)
    noise_power = signal_power / (2 * rate * ebno_linear)
    noise_std_dev = torch.sqrt(noise_power)
    noise = noise_std_dev * torch.randn_like(encoded_data).to(device)
    noisy_data = encoded_data + noise
    return noisy_data

def test_model(encoder, decoder, test_size, batch_size, sequence_length, num_symbols, ebno_db, rate, device):
    encoder.eval()  # Set the encoder to evaluation mode
    decoder.eval()  # Set the decoder to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for inference
        ber = []
        num_batches = int(test_size / batch_size)
        for i in range(num_batches):
            # Select the batch data
            input_data = generate_data(batch_size, sequence_length, num_symbols).to(device)
            encoded_data = encoder(input_data)
            noisy_data = awgn_channel(encoded_data, ebno_db, rate, device)
            decoded_output = decoder(noisy_data)
            ber.append(compute_ber(decoded_output, input_data, mode="binary"))
        logging.info(f"Test BER: {np.mean(ber):.4f}")

def overfit_single_batch(encoder, decoder, device, batch_size, sequence_length, num_symbols, learning_rate, ebno_db, rate, num_iterations=1000,  **kwargs):
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=0.01)
    input_data = generate_data(batch_size, sequence_length, num_symbols).to(device)
    
    loss_log, ber_log = [], []
    encoder.train()
    decoder.train()
    
    def detach_hidden_states(hidden_states):
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states.detach()
        elif isinstance(hidden_states, list) or isinstance(hidden_states, tuple):
            return [detach_hidden_states(h) for h in hidden_states]
        else:
            return hidden_states  # For any other types, return as is
    
    # Helper function to handle encoder forward pass with or without hidden states
    def encoder_forward(inputs, h_0=None, update_hidden=False):
        if update_hidden:
            # Perform forward pass with provided hidden states
            encoded_data, hidden_states = encoder(inputs, *h_0)
            
            # Detach hidden states recursively
            hidden_states = detach_hidden_states(hidden_states)
            return encoded_data, hidden_states
        else:
            # Forward pass without updating hidden states
            encoded_data, hidden_states = encoder(inputs)
            
            # Detach hidden states recursively
            hidden_states = detach_hidden_states(hidden_states)
            return encoded_data, hidden_states

    # Initialize hidden states if the encoder requires it
    h_0 = [None, None]# if getattr(encoder, "enc_rnn_1", None) else None
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        encoded_data = encoder(input_data)
        noisy_data = awgn_channel(encoded_data, ebno_db, rate, device)
        decoded_output = decoder(noisy_data)
        loss = F.binary_cross_entropy(decoded_output, input_data)
        
        loss_log.append(loss.item())
        
        ber = compute_ber(decoded_output, input_data, mode="binary")
        ber_log.append(ber)
        
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            logging.info(f"Iteration [{iteration}/{num_iterations}], Loss: {loss.item():.4f}, BER: {ber:.4e}")
    
    print("Single-batch overfitting finished.")

# Training Loop
def train_model(encoder, decoder, device, epochs, batch_size, sequence_length, num_symbols, learning_rate, ebno_db, rate, sample_size, **kwargs):

    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=0.01)
    for epoch in range(epochs):
        ber, loss_ = [],[]
        encoder.train()
        decoder.train()
        for _ in range(int(sample_size/batch_size)):
            optimizer.zero_grad()
            input_data = generate_data(batch_size, sequence_length, num_symbols).to(device)
            encoded_data = encoder(input_data)
            noisy_data = awgn_channel(encoded_data, ebno_db, rate, device)
            decoded_output = decoder(noisy_data)
            loss = F.binary_cross_entropy(decoded_output, input_data)
            loss_.append(loss.item())
            ber.append(compute_ber(decoded_output, input_data, mode="binary"))
            loss.backward()
            #nn_utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1)
            optimizer.step()
            
        logging.info(f"Epoch [{epoch}/{epochs}], Loss: {np.mean(loss_):.4f}, Ber: {np.mean(ber):.4f}")
        if (epoch + 1) % 10 == 0:
            test_model(encoder, decoder, sample_size, batch_size, sequence_length, num_symbols, ebno_db, rate, device)
        
    print("Training finished.")

def train_model_alternate(encoder, decoder, device, epochs, batch_size, dec_bs_fac, sequence_length, num_symbols, learning_rate, ebno_db, rate, sample_size, **kwargs):
    
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=0.01)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=0.01)

    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    total_batches = int(sample_size / batch_size)
    
    #def encoder_forward(inputs, h_0=None, update_hidden=True):
    #    # Dynamically handle hidden states based on encoder type
    #    if update_hidden and h_0 is not None:
    #        encoded_data, *hidden_states = encoder(inputs, *h_0)
    #        hidden_states = [h.detach() for h in hidden_states]  # Detach to prevent backprop across batches
    #        return encoded_data, hidden_states
    #    else:
    #        return encoder(inputs), None  # Encoder does not require hidden states

    for epoch in range(epochs):
        ber, loss_ = [],[]
        #h_0 = [None, None] if getattr(encoder, "enc_rnn_1", None) else None  # Init hidden states if encoder needs it
        
        for _ in range(total_batches):
            encoder_optimizer.zero_grad()
            input_data = generate_data(batch_size, sequence_length, num_symbols).to(device)
            encoded_data = encoder(input_data)
            noisy_data = awgn_channel(encoded_data, ebno_db, rate, device, decoder_training=False)
            decoded_output = decoder(noisy_data)
            loss = F.binary_cross_entropy(decoded_output, input_data)
            loss_.append(loss.item())
            ber.append(compute_ber(decoded_output, input_data, mode="binary"))
            loss.backward()
            #nn_utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1)
            encoder_optimizer.step()
            
        logging.info(f"Encoder: Epoch [{epoch+1}/{epochs}], Loss: {np.mean(loss_):.6f}, Ber: {np.mean(ber):.6f}")

        ber, loss_ = [], []

        for _ in range(5 * total_batches):
            decoder_optimizer.zero_grad()
            input_data = generate_data(batch_size*dec_bs_fac, sequence_length, num_symbols).to(device)
            encoded_data= encoder(input_data)  # Skip hidden state updates for decoder phase
            encoded_data = encoded_data.detach()
            noisy_data = awgn_channel(encoded_data, ebno_db, rate, device, decoder_training=True)
            decoded_output = decoder(noisy_data)
            loss = F.binary_cross_entropy(decoded_output, input_data)
            loss_.append(loss.item())
            ber.append(compute_ber(decoded_output, input_data, mode="binary"))

            loss.backward()
            decoder_optimizer.step()
            

        logging.info(f"Decoder: Epoch [{epoch+1}/{epochs}], Loss: {np.mean(loss_):.6f}, Ber: {np.mean(ber):.6f}")
        if (epoch + 1) % 50 == 0:
            test_model(encoder, decoder, sample_size, batch_size, sequence_length, num_symbols, ebno_db, rate, device)
    logging.info("Training finished.")
    
    
    ####
    # encoder_optimizer = optim.Shampoo(
    #     encoder.parameters(), 
    #     lr=learning_rate,
    #     momentum=0.0,            # You can adjust this if needed
    #     weight_decay=0.01,       # As per requirement
    #     epsilon=1e-4,            # A small constant for stability
    #     update_freq=1            # Frequency of update for preconditioners
    # )
    # decoder_optimizer = optim.Shampoo(
    #     decoder.parameters(), 
    #     lr=learning_rate,
    #     momentum=0.0,            # You can adjust this if needed
    #     weight_decay=0.01,       # As per requirement
    #     epsilon=1e-4,            # A small constant for stability
    #     update_freq=1            # Frequency of update for preconditioners
    # )