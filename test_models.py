import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from model_turboAE import ENC_CNNTurbo, DEC_CNNTurbo, ENC_CNNTurbo_serial, DEC_CNNTurbo_serial, ENC_GRUTurbo
from model_prod import ProductAEEncoder, ProdDecoder



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

def setup_logging():
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    filename = "test_models.log"
    log_path = os.path.join(log_directory, filename)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def test_model(encoder, decoder, test_size, batch_size, sequence_length, num_symbols, ebno_db, rate, device):
    """Test the model over a specified Eb/N0 and return the BER."""
    encoder.eval()  # Set the encoder to evaluation mode
    decoder.eval()  # Set the decoder to evaluation mode

    ber = []  # Store BER for each batch
    num_batches = int(test_size / batch_size)

    with torch.no_grad():  # Disable gradient calculation for inference
        for i in range(num_batches):
            # Generate input data
            input_data = generate_data(batch_size, sequence_length, num_symbols).to(device)

            # Forward pass through the model
            encoded_data = encoder(input_data)
            noisy_data = awgn_channel(encoded_data, ebno_db, rate, device)
            decoded_output = decoder(noisy_data)

            # Compute BER for the batch
            ber.append(compute_ber(decoded_output, input_data, mode="binary"))

    avg_ber = np.mean(ber)
    logging.info(f"Eb/N0: {ebno_db} dB, Test BER: {avg_ber:.4e}")
    return avg_ber

def load_model(model_path, model_class, device):
    """Load a model from file."""
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def plot_results(ebno_range, results, labels):
    """Plot BER vs. Eb/N0 for multiple models."""
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(results):
        plt.semilogy(ebno_range, result, label=labels[i])
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("BER Performance Comparison")
    plt.show()

def main():
    setup_logging()

    # Configuration
    ebno_range = np.arange(0, 6.5, 0.5)  # Range of Eb/N0 values to test
    test_size = 10000
    batch_size = 500
    sequence_length = 64
    num_symbols = 2
    rate = 64 / 128  # Assume rate as sequence_length / channel_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model paths and labels
    model_configs = [
        {
            "encoder_path": "saved_models/cnn_turbo_encoder.pth",
            "decoder_path": "saved_models/cnn_turbo_decoder.pth",
            "model_type": "cnn_turbo",
            "encoder_class": ENC_CNNTurbo,
            "decoder_class": DEC_CNNTurbo
        },
        {
            "encoder_path": "saved_models/CNN_turbo_serial_encoder.pth",
            "decoder_path": "saved_models/CNN_turbo_serial_decoder.pth",
            "model_type": "CNN_turbo_serial",
            "encoder_class": ENC_CNNTurbo_serial,
            "decoder_class": DEC_CNNTurbo_serial
        },
        {
            "encoder_path": "saved_models/Product_AE_encoder.pth",
            "decoder_path": "saved_models/Product_AE_decoder.pth",
            "model_type": "Product_AE",
            "encoder_class": ProductAEEncoder,
            "decoder_class": ProdDecoder
        },
        {
            "encoder_path": "saved_models/CNN_turbo_serial_encoder.pth",
            "decoder_path": "saved_models/CNN_turbo_serial_decoder.pth",
            "model_type": "CNN_turbo_serial",
            "encoder_class": ENC_GRUTurbo,
            "decoder_class": DEC_CNNTurbo_serial
        },
    ]

    results = []
    labels = []

    for config in model_configs:
        logging.info(f"Testing model: {config['model_type']}")
        
        # Load encoder and decoder
        encoder = load_model(config['encoder_path'], config['encoder_class'], device)
        decoder = load_model(config['decoder_path'], config['decoder_class'], device)

        # Test over the Eb/N0 range
        ber_result = []
        for ebno_db in ebno_range:
            ber_result.append(test_model(encoder, decoder, test_size, batch_size,
                                         sequence_length, num_symbols, ebno_db, rate, device))
        
        results.append(ber_result)
        labels.append(config['model_type'])

    # Plot results
    plot_results(ebno_range, results, labels)

if __name__ == '__main__':
    main()
