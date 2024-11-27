import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from model_turboAE import TurboConfig, Interleaver, ENC_CNNTurbo, DEC_CNNTurbo, ENC_CNNTurbo_serial, DEC_CNNTurbo_serial, ENC_GRUTurbo
from model_prod import ProductAEEncoder, ProdDecoder

# Data Generation Function
def generate_data(batch_size, sequence_length, num_symbols):
    return torch.randint(0, num_symbols, (batch_size, sequence_length), dtype=torch.float)

def compute_bler(decoded_output, inputs, mode="symbol", num_symbols=None):
    """
    Calculate the Block Error Rate (BLER).
    """
    if mode == "symbol":
        if num_symbols is None:
            raise ValueError("num_symbols must be specified when mode is 'symbol'")
        
        # Get predicted symbols from softmax or one-hot encoded outputs
        predicted_symbols = torch.argmax(decoded_output, dim=-1)
        # Calculate block errors (any symbol error in a sequence marks the block as erroneous)
        block_errors = (predicted_symbols != inputs).any(dim=1).float()
        # Calculate BLER
        BLER = torch.mean(block_errors).detach().cpu().item()

    elif mode == "binary":
        # Calculate BLER for binary (sigmoid) outputs
        binary_predictions = torch.round(decoded_output)
        # Check if there is at least one error in each block
        block_errors = torch.any(binary_predictions != inputs, dim=1).float()
        # Calculate BLER as the mean of block errors
        BLER = torch.mean(block_errors).detach().cpu().item()
    else:
        raise ValueError("Unsupported mode. Choose 'symbol' or 'binary'.")
    
    return BLER

def compute_ber(decoded_output, inputs, num_symbols=None, mode="symbol"):
    """
    Calculate the Bit Error Rate (BER) for different types of decoded outputs.
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

# Main functions

def setup_logging():
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_directory, "test_all_models.log"), 
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def test_model(encoder, decoder, config, ebno_db):
    """
    Test the model over a specified Eb/N0 and return BER and BLER.
    """
    encoder.eval()
    decoder.eval()
    ber_list = []
    bler_list = []
    num_batches = int(config['test_size'] / config['batch_size'])

    with torch.no_grad():
        for _ in range(num_batches):
            input_data = generate_data(config['batch_size'], config['sequence_length'], config['num_symbols']).to(config['device'])
            encoded_data = encoder(input_data)
            noisy_data = awgn_channel(encoded_data, ebno_db, config['rate'], config['device'])
            decoded_output = decoder(noisy_data)
            
            # Compute BER and BLER
            ber_list.append(compute_ber(decoded_output, input_data, mode="binary"))
            bler_list.append(compute_bler(decoded_output, input_data, mode="binary"))

    avg_ber = np.mean(ber_list)
    avg_bler = np.mean(bler_list)
    logging.info(f"Eb/N0: {ebno_db} dB, Test BER: {avg_ber:.4e}, Test BLER: {avg_bler:.4e}")
    return avg_ber, avg_bler

def plot_ber_results(ebno_range, ber_results, labels, save_path="results/ber_comparison.png"):
    """
    Plot BER results.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for i, ber in enumerate(ber_results):
        plt.semilogy(ebno_range, ber, label=labels[i], linestyle="--")
    
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("BER Performance Comparison")
    plt.savefig(save_path, bbox_inches="tight")
    logging.info(f"BER plot saved to {save_path}")
    plt.close()

def plot_bler_results(ebno_range, bler_results, labels, save_path="results/bler_comparison.png"):
    """
    Plot BLER results.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for i, bler in enumerate(bler_results):
        plt.semilogy(ebno_range, bler, label=labels[i], linestyle="-")
    
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Block Error Rate (BLER)")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("BLER Performance Comparison")
    plt.savefig(save_path, bbox_inches="tight")
    logging.info(f"BLER plot saved to {save_path}")
    plt.close()

def main():
    setup_logging()

    # Configuration
    config = {
        'ebno_range': np.arange(0, 6.5, 0.5),
        'test_size': 10000,
        'batch_size': 500,
        'sequence_length': 64,
        'num_symbols': 2,
        'rate': 64 / 128,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # List of models to test
    models = [
        {
            'name': 'cnn_turbo',
            'init': lambda: {
                'config_params': (config_params := TurboConfig(
                    block_len=config['sequence_length'],
                    enc_num_unit=100,
                    dec_num_unit=100,
                    batch_size=config['batch_size'],
                    enc_num_layer=config['enc_num_layers'],
                    dec_num_layer=config['dec_num_layers']
                )),
                'interleaver': (interleaver := Interleaver(config_params).to(config['device'])),
                'encoder': lambda: (
                    model := ENC_CNNTurbo(config_params, interleaver).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/cnn_turbo_encoder.pth", map_location=config['device'])),
                    model
                )[-1],
                'decoder': lambda: (
                    model := DEC_CNNTurbo(config_params, interleaver).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/cnn_turbo_decoder.pth", map_location=config['device'])),
                    model
                )[-1]
            }
        },
        {
            'name': 'CNN_turbo_serial',
            'init': lambda: {
                'config_params': (config_params := TurboConfig(
                    block_len=config['sequence_length'],
                    enc_num_unit=100,
                    dec_num_unit=100,
                    batch_size=config['batch_size'],
                    enc_num_layer= config['enc_num_layers'],
                    dec_num_layer=config['dec_num_layers']
                )),
                'interleaver': (interleaver := Interleaver(config_params).to(config['device'])),
                'encoder': lambda: (
                    model := ENC_CNNTurbo_serial(config_params, interleaver).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/CNN_turbo_serial_encoder.pth", map_location=config['device'])),
                    model
                )[-1],
                'decoder': lambda: (
                    model := DEC_CNNTurbo_serial(config_params, interleaver).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/CNN_turbo_serial_decoder.pth", map_location=config['device'])),
                    model
                )[-1]
            }
        },
        {
            'name': 'Product_AE',
            'init': lambda: {
                'K': (K := [8, 8]),  # Dimensions for Product_AE
                'N': (N := [8, 16]),  # Dimensions for Product_AE
                'I': (I := 4),  # Iterations for decoding
                'encoder': lambda: (
                    model := ProductAEEncoder(K, N).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/Product_AE_encoder.pth", map_location=config['device'])),
                    model
                )[-1],
                'decoder': lambda: (
                    model := ProdDecoder(I, K, N).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/Product_AE_decoder.pth", map_location=config['device'])),
                    model
                )[-1]
            }
        },
        {
            'name': 'gru_turbo',
            'init': lambda: {
                'config_params': (config_params := TurboConfig(
                    block_len=config['sequence_length'],
                    enc_num_unit=100,
                    dec_num_unit=100,
                    batch_size=config['batch_size'],
                    enc_num_layer= config['enc_num_layers'],
                    hidden_size_gru=config['hidden_size_gru'],
                    dec_num_layer=config['dec_num_layers']
                )),
                'interleaver': (interleaver := Interleaver(config_params).to(config['device'])),
                'encoder': lambda: (
                    model := ENC_GRUTurbo(config_params, interleaver).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/gru_turbo_encoder.pth", map_location=config['device'])),
                    model
                )[-1],
                'decoder': lambda: (
                    model := DEC_CNNTurbo(config_params, interleaver).to(config['device']),
                    model.load_state_dict(torch.load("saved_models/gru_turbo_decoder.pth", map_location=config['device'])),
                    model
                )[-1]
            }
        }
    ]


    results = []
    labels = []

    for model in models:
        logging.info(f"Testing model: {model['name']}")
        # Initialize the model components
        init = model['init']()

        encoder = init['encoder']()
        decoder = init['decoder']()

        # Test over the Eb/N0 range
        ber_result = []
        for ebno_db in config['ebno_range']:
            ber_result.append(test_model(encoder, decoder, config, ebno_db))
        results.append(ber_result)
        labels.append(model['name'])

    # Plot and save results
    plot_ber_results(config['ebno_range'], results, labels, save_path="results/ber_comparison.png")
    plot_bler_results(config['ebno_range'], results, labels, save_path="results/ber_comparison.png")

if __name__ == '__main__':
    main()

