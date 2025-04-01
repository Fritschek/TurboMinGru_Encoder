import re
import os
import matplotlib.pyplot as plt

# Function to parse log files for epochs
def parse_epoch_log_file(log_file):
    model_type = None
    hidden_size_gru = None
    epochs = []
    losses = []
    bers = []

    # Regex patterns for parsing
    model_pattern = r"Configuration:.*'model_type':\s*'([\w_]+)'"
    hidden_size_pattern = r"'hidden_size_gru':\s*(\d+)"
    epoch_pattern = r"Epoch \[(\d+)/\d+\], Loss: ([0-9.]+), Ber: ([0-9.e+-]+)"

    with open(log_file, 'r') as file:
        for line in file:
            # Extract model type
            if model_type is None:
                model_match = re.search(model_pattern, line)
                if model_match:
                    model_type = model_match.group(1)

            # Extract hidden size
            if hidden_size_gru is None:
                hidden_size_match = re.search(hidden_size_pattern, line)
                if hidden_size_match:
                    hidden_size_gru = int(hidden_size_match.group(1))

            # Extract epoch, loss, and BER
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
                losses.append(float(epoch_match.group(2)))
                bers.append(float(epoch_match.group(3)))

    return model_type, hidden_size_gru, epochs, losses, bers

# Function to plot loss and BER for encoder/decoder with alternating pattern and save as PDF
def plot_loss_and_ber(log_files, output_dir="plots", y_min=1e-4, y_max=1e0):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Initialize storage for plots
    all_model_labels = []
    encoder_epochs = []
    encoder_losses = []
    encoder_bers = []
    decoder_epochs = []
    decoder_losses = []
    decoder_bers = []

    for log_file in log_files:
        model_type, hidden_size_gru, epochs, losses, bers = parse_epoch_log_file(log_file)
        if model_type is not None:
            label = f'{model_type} (hidden_size={hidden_size_gru})'
            all_model_labels.append(label)

            # Separate encoder and decoder data
            enc_epochs = [epochs[i] for i in range(0, len(epochs), 2)]
            dec_epochs = [epochs[i] for i in range(1, len(epochs), 2)]

            enc_losses = [losses[i] for i in range(0, len(losses), 2)]
            dec_losses = [losses[i] for i in range(1, len(losses), 2)]

            enc_bers = [bers[i] for i in range(0, len(bers), 2)]
            dec_bers = [bers[i] for i in range(1, len(bers), 2)]

            # Store raw data
            encoder_epochs.append(enc_epochs)
            encoder_losses.append(enc_losses)
            encoder_bers.append(enc_bers)

            decoder_epochs.append(dec_epochs)
            decoder_losses.append(dec_losses)
            decoder_bers.append(dec_bers)
        else:
            print(f"Warning: No model_type found in {log_file}. Skipping file.")

    # Plot and save Encoder Loss
    plt.figure()
    for i in range(len(all_model_labels)):
        plt.plot(encoder_epochs[i], encoder_losses[i], label=f'{all_model_labels[i]}', alpha=1.0)
    plt.xlabel('Epochs', fontsize=12, weight='bold')
    plt.ylabel('Loss', fontsize=12, weight='bold')
    plt.yscale('log')
    plt.ylim(y_min, y_max)  # Set fixed y-axis range
    plt.title('Encoder Training Loss', fontsize=14, weight='bold')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f"{output_dir}/encoder_loss.pdf",bbox_inches="tight", dpi=300)  # Save as PDF
    plt.show()

    # Plot and save Encoder BER
    plt.figure()
    for i in range(len(all_model_labels)):
        plt.plot(encoder_epochs[i], encoder_bers[i], label=f'{all_model_labels[i]}', alpha=1.0)
    plt.xlabel('Epochs', fontsize=12, weight = 'bold')
    plt.ylabel('BER', fontsize=12, weight='bold')
    plt.yscale('log')
    plt.ylim(y_min, y_max)  # Set fixed y-axis range
    plt.title('Encoder Training BER', fontsize=14, weight='bold')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f"{output_dir}/encoder_ber.pdf", bbox_inches="tight", dpi=300)  # Save as PDF
    plt.show()

    # Plot and save Decoder Loss
    plt.figure()
    for i in range(len(all_model_labels)):
        plt.plot(decoder_epochs[i], decoder_losses[i], label=f'{all_model_labels[i]}', alpha=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.ylim(y_min, y_max)  # Set fixed y-axis range
    plt.title('Decoder Training Loss')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f"{output_dir}/decoder_loss.pdf")  # Save as PDF
    plt.show()

    # Plot and save Decoder BER
    plt.figure()
    for i in range(len(all_model_labels)):
        plt.plot(decoder_epochs[i], decoder_bers[i], label=f'{all_model_labels[i]}', alpha=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.ylim(y_min, y_max)  # Set fixed y-axis range
    plt.title('Decoder Training BER')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f"{output_dir}/decoder_ber.pdf")  # Save as PDF
    plt.show()

# Main function
def main():
    # List of log files (replace with actual file names)
    log_files = [
        'training_gru_turbo_2024-12-09_09-33-58_7474.log',
        'training_gru_turbo_2024-12-09_09-33-58_8926.log',
        'training_gru_turbo_2024-12-09_09-33-59_2955.log',
        'training_gru_turbo_2024-12-09_09-35-30_1396.log',
        'training_gru_turbo_2024-12-09_09-35-30_3063.log',
        'training_gru_turbo_2024-12-09_09-35-30_8882.log',
        'training_gru_turbo_2024-12-09_09-35-30_9532.log',
        'training_gru_turbo_2024-12-09_09-59-23_7689.log',
        'training_gru_turbo_2024-12-09_10-09-00_8867.log',
        'training_gru_turbo_2024-12-09_10-17-58_1887.log'
        #'training_gru_turbo_2024-12-09_10-31-30_8642.log'
    ]
    plot_loss_and_ber(log_files, output_dir="results", y_min=None, y_max=3e-2)

if __name__ == "__main__":
    main()
