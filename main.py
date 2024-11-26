import logging
from datetime import datetime
import torch
import numpy as np
import argparse
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
# Parse arguments
parser = argparse.ArgumentParser(description="Experiment Configuration")
parser.add_argument('--model_type', type=str, default='gru_turbo', help="Model type: 'cnn_turbo';'gru_turbo' or 'CNN_turbo_serial")
parser.add_argument('--num_symbols', type=int, default=2, help="Number of Symbols (2 for bits)")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--dec_bs_fac', type=int, default=4, help="Decoder Batch Size Factor")
parser.add_argument('--sample_size', type=int, default=5000, help="Number of samples")
parser.add_argument('--sequence_length', type=int, default=64, help="Sequence length")
parser.add_argument('--channel_length', type=int, default=128, help="Channel length")
parser.add_argument('--epochs', type=int, default=500, help="Number of epochs")
parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate")
parser.add_argument('--ebno_db', type=float, default=4, help="Eb/N0 in dB")
parser.add_argument('--enc_num_layers', type=int, default=2, help="Number of layers in the encoder")
parser.add_argument('--dec_num_layers', type=int, default=5, help="Number of layers in the encoder")
parser.add_argument('--hidden_size_gru', type=int, default=4, help="Hidden size of GRU")

args = parser.parse_args()

# Configurations
config = {
    'model_type': args.model_type,
    'num_symbols': args.num_symbols,
    'batch_size': args.batch_size,
    'dec_bs_fac': args.dec_bs_fac,
    'sample_size': args.sample_size,
    'sequence_length': args.sequence_length,
    'channel_length': args.channel_length,
    'rate': args.sequence_length / args.channel_length,
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'ebno_db': args.ebno_db,
    'enc_num_layers': args.enc_num_layers,
    'dec_num_layers': args.dec_num_layers,
    'hidden_size_gru': args.hidden_size_gru
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
                                              batch_size=config['batch_size'],
                                              enc_num_layer=config['enc_num_layers'],
                                              dec_num_layer=config['dec_num_layers'])
        interleaver = model_TAE.Interleaver(config_params)
        encoder = model_TAE.ENC_CNNTurbo(config_params, interleaver).to(device)
        decoder = model_TAE.DEC_CNNTurbo(config_params, interleaver).to(device)
        
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
                                              batch_size=config['batch_size'],
                                              enc_num_layer= config['enc_num_layers'],
                                              dec_num_layer=config['dec_num_layers']
                                            )
        interleaver = model_TAE.Interleaver(config_params).to(device)
        encoder = model_TAE.ENC_CNNTurbo_serial(config_params, interleaver).to(device)
        decoder = model_TAE.DEC_CNNTurbo_serial(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_LargeMinRNNTurbo(config_params, interleaver).to(device)
        #decoder = model_TAE.DEC_GRUTurbo_test(config_params, interleaver).to(device)
        
    elif config['model_type'] == 'gru_turbo':
        config_params = model_TAE.TurboConfig(block_len=config['sequence_length'],
                                              enc_num_unit=100,
                                              dec_num_unit=100,
                                              batch_size=config['batch_size'],
                                              enc_num_layer= config['enc_num_layers'],
                                              hidden_size_gru=config['hidden_size_gru'],
                                              dec_num_layer=config['dec_num_layers']
                                              )
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
    
    train.train_model_alternate(encoder, decoder, device, **config)
    #train.overfit_single_batch(encoder, decoder, device, **config)
    
    save_models(encoder, decoder, config['model_type'])
    
def save_models(encoder, decoder, model_type):
    """Saves the encoder and decoder models without overriding existing files."""
    model_save_dir = "saved_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Generate unique filenames if files already exist
    encoder_save_path = os.path.join(model_save_dir, f"{model_type}_encoder.pth")
    decoder_save_path = os.path.join(model_save_dir, f"{model_type}_decoder.pth")

    if os.path.exists(encoder_save_path) or os.path.exists(decoder_save_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_save_path = os.path.join(model_save_dir, f"{model_type}_encoder_{timestamp}.pth")
        decoder_save_path = os.path.join(model_save_dir, f"{model_type}_decoder_{timestamp}.pth")

    # Save the models
    torch.save(encoder.state_dict(), encoder_save_path)
    torch.save(decoder.state_dict(), decoder_save_path)

    logging.info(f"Encoder saved to {encoder_save_path}")
    logging.info(f"Decoder saved to {decoder_save_path}")
    
    
if __name__ == '__main__':
    main()