# BitNet - Binary Neural Network for Time-Series Forecasting

A C implementation of a binary neural network inspired by the **BitNet** architecture, designed for solar X-ray flux prediction using GOES data.

## Overview

This project implements a binary (1-bit) neural network with bias terms and momentum, applying the core principles of BitNet's quantization approach to time-series forecasting. Unlike traditional floating-point networks, this implementation uses binary weights (+1/-1) for both forward propagation and backpropagation, demonstrating the efficiency benefits of 1-bit computation.

### Inspiration: BitNet Architecture

This implementation draws from [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) (Wang et al., 2023), which introduced a scalable 1-bit Transformer architecture for LLMs. Key concepts adapted here include:

- **Binary weight quantization**: Mapping floating-point weights to binary values (+1/-1)
- **XOR-based computation**: Using binary weights in dot products during inference
- **Quantization-aware updates**: Re-quantizing weights after gradient updates

For the complete BitNet b1.58 technical report and modern implementation, see [microsoft/BitNet](https://github.com/microsoft/BitNet).

## Features

- Binary (1-bit) weight representation: `+1` or `-1`
- Momentum-based weight updates for stable training
- Bias terms in all layers
- Early stopping based on test error threshold
- Open-loop and closed-loop prediction evaluation
- NMSE (Normalized Mean Squared Error) reporting

## Architecture

| Parameter | Value |
|-----------|-------|
| Input units | 20 (N) |
| Hidden layer | 16 units |
| Output units | 1 (M) |
| Activation | Sigmoid |
| Weight type | Binary (+1/-1) |
| Learning rate (η) | 0.25 |
| Momentum (α) | 0.9 |
| Gain | 1.0 |

## Data

The network is trained on **GOES X-ray flux data** (normalized to [0.1, 0.9]):

- **Training set**: Years 2131–2150 (180 time steps)
- **Test set**: Years 2151–2200 (50 time steps)
- **Evaluation set**: Years 2201–2230 (29 time steps)

## Building and Running

### Prerequisites

- C compiler (gcc, clang)
- Standard math library (`libm`)

### Build

```bash
make
```

This compiles `bitnet.c` into the `bitnet` executable.

### Run

```bash
./bitnet
```

### Output

The program writes results to `BitNet.txt`:

```
NMSE is 0.167 on Training Set and 0.283 on Test Set - stopping Training and restoring Weights ...

Year    Solar Flux    Open-Loop Prediction    Closed-Loop Prediction
...
```

## Project Structure

| File | Description |
|------|-------------|
| `bitnet.c` | Main source code (binary NN implementation) |
| `bpn.c` | Original Backpropagation Neural Network reference |
| `Makefile` | Build configuration |
| `BitNet.txt` | Training output and evaluation results |
| `README.md` | This file |
| `design.md` | Architecture design documentation |
| `reference.md` | BitNet paper background and references |

## License

This implementation is for educational purposes. The original code was adapted from BPN.c by Karsten Kutza (1996).

## References

1. Wang, H., Ma, S., Dong, L., et al. "BitNet: Scaling 1-bit Transformers for Large Language Models." *arXiv:2310.11453*, 2023.
2. Microsoft. "bitnet.cpp: Official Inference Framework for 1-bit LLMs." *GitHub*, 2025. [https://github.com/microsoft/BitNet](https://github.com/microsoft/BitNet)
3. Ma, S., Wang, H., Huang, S., et al. "BitNet b1.58 2B4T Technical Report." *arXiv:2504.12285*, 2025.
