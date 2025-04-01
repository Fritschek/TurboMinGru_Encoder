# MinGRU-Based Encoder for Turbo Autoencoder Frameworks

This repository contains the official implementation for the paper:

> **MinGRU-Based Encoder for Turbo Autoencoder Frameworks**  
> *Rick Fritschek, Rafael F. Schaefer*  
> arXiv preprint [arXiv:2503.08451](https://arxiv.org/abs/2503.08451), March 2025

---

## Overview

Early neural channel coding approaches used dense neural networks with one-hot encodings, which performed well for small block lengths but failed to scale. TurboAE addressed this with bit-sequence inputs and convolutional encoder-decoder blocks, improving scalability at the expense of sequential modeling capability.

In this work, we revisit RNNs for Turbo autoencoders, leveraging efficient sequence models—**minGRU** and **Mamba blocks**—to construct a lightweight yet scalable **GRU-based Turbo autoencoder**. Our architecture:

- Matches or outperforms CNN-based TurboAE for short sequences.
- Scales better to long sequences.
- Reduces training time and memory footprint.

---

## Citation

If you use this code or build upon our work, please cite:

```bibtex
@article{fritschek2025mingru,
  title={MinGRU-Based Encoder for Turbo Autoencoder Frameworks},
  author={Rick Fritschek and Rafael F. Schaefer},
  journal={arXiv preprint arXiv:2503.08451},
  year={2025}
}
```

---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.1  
- CUDA 12.1 (tested on NVIDIA H100)

---

## Running the Model

To train the main **GRU-based Turbo Autoencoder**, run:

```bash
python main.py \
  --model_type gru_turbo \
  --enc_num_layers 2 \
  --hidden_size_gru 4 \
  --sequence_length 64 \
  --channel_length 128 \
  --epochs 500 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --ebno_db 4
```

> 💡 **Note**: On an NVIDIA H100 GPU, a full training run takes approximately **7 hours**.

---

## Configuration Parameters

| Argument            | Description                             | Default |
|---------------------|-----------------------------------------|---------|
| `--model_type`      | Model architecture (`gru_turbo`)        | `gru_turbo` |
| `--enc_num_layers`  | Number of encoder GRU layers            | `2`     |
| `--dec_num_layers`  | Number of decoder layers                | `5`     |
| `--hidden_size_gru` | GRU hidden size                         | `4`     |
| `--sequence_length` | Message sequence length (input bits)    | `64`    |
| `--channel_length`  | Channel length (transmission bits)      | `128`   |
| `--epochs`          | Number of training epochs               | `500`   |
| `--ebno_db`         | Signal-to-noise ratio in dB             | `4`     |
| `--sample_size`      | Number of training samples             | `50000` |

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or collaborations, please contact:

- **Rick Fritschek** – [rick.fritschek@tu-dresden.de](mailto:rick.fritschek@tu-dresden.de)  
- Chair of Information Theory and Machine Learning, TU Dresden
