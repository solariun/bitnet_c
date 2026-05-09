# BitNet: Binary Neural Network for Time-Series Forecasting

This project implements a Binary Neural Network (BitNet) for predicting Solar X-ray Flux using binary (1-bit) weights instead of floating-point weights. The implementation is inspired by the BitNet architecture introduced in the paper "BitNet: Scaling 1-bit Transformers for Large Language Models" (arXiv:2310.11453).

## Overview

BitNet uses binary weights (+1 or -1) throughout the network, significantly reducing memory footprint and computational requirements while maintaining competitive performance. This implementation demonstrates the BitNet principles applied to time-series forecasting.

## Architecture

The network consists of three layers:
- **Input Layer**: 20 units (time series window)
- **Hidden Layer**: 16 units with sigmoid activation
- **Output Layer**: 1 unit (forecasted value)

Key features:
- Binary quantization of weights (+1/-1)
- Momentum-based gradient descent
- Bias terms for each layer
- Early stopping based on validation performance

## Installation

```bash
make clean && make
```

## Usage

Run the training and evaluation:

```bash
./bitnet
```

The program will:
1. Initialize random weights
2. Train the network with early stopping
3. Evaluate on training and test sets
4. Output predictions to `BitNet.txt`

## Input Data

The project uses Solar X-ray Flux data from NOAA/NCEI GOES archives (0.1-0.8 nm band). The data spans 260 years and is normalized to the range [0.1, 0.9].

## Output Format

The `BitNet.txt` file contains:
- Training progress with NMSE values
- Final evaluation metrics
- Open-loop and closed-loop predictions for future years

Example output format:
```
Year    Solar Flux    Open-Loop Prediction    Closed-Loop Prediction
2131       0.100                   0.174                     0.174
```

## Comparison with BPN

This repository includes both `bitnet.c` (Binary Network) and `bpn.c` (Backpropagation Network). The key differences:

| Feature | BitNet | BPN |
|---------|--------|-----|
| Weights | Binary (1-bit) | Floating-point |
| Memory | 32× less | Standard |
| Computation | Sign-based | Multiplication |

## Files

- `bitnet.c` - BitNet implementation with binary weights
- `bpn.c` - Reference Backpropagation Network implementation
- `Makefile` - Build configuration
- `design.md` - Technical design documentation
- `reference.md` - Background on BitNet and the original paper
- `BitNet.txt` - Training output and predictions

## Technical Details

### Binary Quantization

Weights are quantized using sign-based quantization:
```
binary_weight = +1 if weight >= 0 else -1
```

During forward pass, binary weights are used directly. During backpropagation, gradients use binary weights but updates go to floating-point weights.

### Training Parameters

- Learning rate (η): 0.25
- Momentum factor (α): 0.9
- Sigmoid gain: 1.0
- Normalization range: [0.1, 0.9]
- Early stopping threshold: 1.2× minimum test error

## License

This project is provided for educational purposes.

## References

- Wang, H., Ma, S., Dong, L., et al. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models*. arXiv:2310.11453
- Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). *Learning Internal Representations by Error Propagation*.

## Author

Modified from BPN.c by Karsten Kutza (17.4.96)