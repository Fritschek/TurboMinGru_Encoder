import logging
from datetime import datetime
import torch
import numpy as np
import os

# Seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Own libraries
import train as train
import model_turboAE as model_TAE
import model_prod as model_prod

# Configurations
config = {
    'model_type': 'Product_AE',  # Options: 'cnn_turbo', 'gru_turbo'
    'num_symbols': 2,
    'd_model': 128, # Dimension of model
    'nhead': 8, # Number of attention heads
    'num_layers': 3, # Number of transformer layers
    'dim_feedforward': 512, # Feedforward dimension in transformer
    'batch_size': 500,
    'sample_size': 5000,
    'sequence_length': 64,
    'channel_length': 128,
    'rate': 64 / 128,  # sequence_length / channel_length
    'epochs': 500,
    'learning_rate': 2e-4,
    'ebno_db': 4
}


def setup_logging():
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'training_RS_NET_{date_str}.log'
    log_path = os.path.join(log_directory, filename)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info("Configuration: %s", config)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    setup_logging()

    if config['model_type'] == 'cnn_turbo':
        config_params = model_TAE.TurboConfig(block_len=config['sequence_length'],
                                              enc_num_unit=100,
                                              dec_num_unit=100,
                                              batch_size=config['batch_size'])
        interleaver = model_TAE.Interleaver(config_params)
        encoder = model_TAE.ENC_CNNTurbo(config_params, interleaver)
        decoder = model_TAE.DEC_CNNTurbo(config_params, interleaver)
        
    elif config['model_type'] == 'Product_AE':
        K = [8, 8] # 8*8 = 64
        N = [8, 16] # 8*16 = 128
        I = 4
        encoder = model_prod.ProductAEEncoder(K, N).to(device)
        decoder = model_prod.ProdDecoder(I, K, N).to(device)
        
    elif config['model_type'] == 'rnn_turbo':
        config_params = model_TAE.TurboConfig(block_len=config['sequence_length'],
                                              enc_num_unit=100,
                                              dec_num_unit=100,
                                              batch_size=config['batch_size'])
        interleaver = model_TAE.Interleaver(config_params).to(device)
        encoder = model_TAE.ENC_rnn_rate2(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_CNNTurbo_serial(config_params, interleaver).to(device)
        decoder = model_TAE.DEC_LargeMinRNNTurbo(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_GRUTurbo_test(config_params, interleaver).to(device)
        
    elif config['model_type'] == 'CNN_turbo_serial':
        config_params = model_TAE.TurboConfig(block_len=config['sequence_length'],
                                              enc_num_unit=100,
                                              dec_num_unit=100,
                                              batch_size=config['batch_size'])
        interleaver = model_TAE.Interleaver(config_params).to(device)
        encoder = model_TAE.ENC_CNNTurbo_serial(config_params, interleaver).to(device)
        decoder = model_TAE.DEC_CNNTurbo_serial(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_LargeMinRNNTurbo(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_GRUTurbo_test(config_params, interleaver).to(device)
        
    elif config['model_type'] == 'gru_turbo':
        config_params = model_TAE.TurboConfig(block_len=config['sequence_length'],
                                              enc_num_unit=100,
                                              dec_num_unit=100,
                                              batch_size=config['batch_size'])
        interleaver = model_TAE.Interleaver(config_params).to(device)
        encoder = model_TAE.ENC_GRUTurbo(config_params, interleaver).to(device)
        decoder = model_TAE.DEC_CNNTurbo(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_LargeMinRNNTurbo(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_GRUTurbo_test(config_params, interleaver).to(device)
                                       

    NN_size = count_parameters(encoder) + count_parameters(decoder)
    logging.info(f"Size of the network: {NN_size} parameters")
    logging.info(f"Start Training using {config['model_type']} model")
    logging.info(f"Using CUDA: {torch.cuda.is_available()} on {torch.cuda.device_count()} device(s)" if torch.cuda.is_available() else "Running on CPU")
    
    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)
    
    #train.train_model_alternate(encoder, decoder, **config)
    train.overfit_single_batch(encoder, decoder, device, **config)
    
    save_models(encoder, decoder, config['model_type'])
    
def save_models(encoder, decoder, model_type):
    """Saves the encoder and decoder models."""
    model_save_dir = "saved_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # File paths for saving
    encoder_save_path = os.path.join(model_save_dir, f"{model_type}_encoder.pth")
    decoder_save_path = os.path.join(model_save_dir, f"{model_type}_decoder.pth")

    # Save the models
    torch.save(encoder.state_dict(), encoder_save_path)
    torch.save(decoder.state_dict(), decoder_save_path)

    logging.info(f"Encoder saved to {encoder_save_path}")
    logging.info(f"Decoder saved to {decoder_save_path}")
    
    
if __name__ == '__main__':
    main()