# BitNet Design Document

## Overview

This project implements a Binary Neural Network (BitNet) for time-series forecasting, specifically predicting Solar X-ray Flux data from NOAA/NCEI GOES archives. The implementation is inspired by the BitNet architecture introduced in the paper "BitNet: Scaling 1-bit Transformers for Large Language Models" (arXiv:2310.11453).

## Architecture

### Network Structure

The BitNet implementation uses a feedforward neural network with the following structure:

- **Input Layer**: 20 units (N = 20) representing the input time series window
- **Hidden Layer**: 16 units with sigmoid activation
- **Output Layer**: 1 unit producing the forecasted value

### Key Components

#### Binary Quantization

The core innovation of BitNet is the use of binary (1-bit) weights instead of floating-point weights. This significantly reduces memory footprint and computational requirements.

**Binary Quantization Process**:
```
Weight → Sign(Weight) → {+1, -1}
```

In the forward pass, binary weights are used directly. During backpropagation, gradients are computed using the binary weights but updates are applied to the underlying floating-point weights.

#### Momentum-Based Training

The network uses momentum-based gradient descent with:
- Learning rate (η): 0.25
- Momentum factor (α): 0.9

#### Bias Terms

Each layer includes a bias unit with constant output of 1.0, enabling the network to learn offset values.

## Algorithm

### Forward Pass

1. Set input values to the input layer
2. For each layer (except output):
   - Compute weighted sum using binary weights
   - Apply sigmoid activation: `output = 1 / (1 + exp(-gain * sum))`

### Backward Pass

1. Compute output error: `error = target - output`
2. Calculate output layer errors: `δ = gain * output * (1 - output) * error`
3. Backpropagate through hidden layers using binary weights
4. Update floating-point weights using gradients and momentum

### Weight Update Rule

```
weight[i][j] += eta * error[i] * output[j] + alpha * delta_weight[i][j]
delta_weight[i][j] = eta * error[i] * output[j]
```

After each update, weights are re-quantized to binary values.

## Training Strategy

### Data Preparation

The solar flux data is normalized to the range [0.1, 0.9] using min-max normalization:

```
normalized = ((value - min) / (max - min)) * (HI - LO) + LO
```

where HI = 0.9 and LO = 0.1.

### Training Process

1. **Epoch-based training**: Each epoch processes TRAIN_YEARS (180) random samples
2. **Early stopping**: Training stops when test error exceeds 1.2× minimum test error
3. **Weight saving**: Best weights are saved during training for restoration

### Evaluation Metrics

- **NMSE (Normalized Mean Squared Error)**: Compares prediction error to baseline (mean prediction)
- **Open-loop prediction**: Direct forecast from network
- **Closed-loop prediction**: Network uses its own predictions as input for subsequent steps

## Implementation Details

### Data Structures

```c
typedef struct {
    INT           Units;
    REAL*         Output;      // Layer outputs including bias
    REAL*         Error;       // Error terms for each unit
    REAL**        Weight;      // Floating-point weights (training)
    REAL**        WeightSave;  // Saved weights for early stopping
    REAL**        dWeight;     // Momentum terms
    INT**         BWeight;     // Binary quantized weights
} LAYER;
```

### Application-Specific Constants

- **NUM_LAYERS**: 3 (input, hidden, output)
- **N**: 20 (input window size)
- **M**: 1 (output dimension)
- **TRAIN_YEARS**: 180 (training samples)
- **TEST_YEARS**: 50 (test samples)
- **EVAL_YEARS**: 29 (evaluation samples)

### Random Initialization

Weights are initialized randomly in the range [-0.5, 0.5] using a uniform distribution.

## Comparison with BPN

The BitNet implementation differs from the standard Backpropagation Network (BPN) in several key ways:

| Aspect | BitNet | BPN |
|--------|--------|-----|
| Weights | Binary (1-bit) | Floating-point |
| Forward pass | Uses binary weights | Uses floating-point weights |
| Memory usage | 32× less (per weight) | Standard |
| Quantization | Explicit quantization layer | N/A |

Both networks use the same backpropagation algorithm and training strategy, but BitNet's binary weights enable more efficient computation.

## File Structure

```
.
├── bitnet.c      # BitNet implementation
├── bpn.c         # Backpropagation Network (reference)
├── Makefile      # Build configuration
├── BitNet.txt    # Training output and predictions
└── README.md     # Project documentation
```

## Usage

```bash
make clean && make
./bitnet
```

The program trains the network, evaluates performance, and outputs predictions to `BitNet.txt`.
