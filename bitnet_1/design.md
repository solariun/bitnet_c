# BitNet Binary Neural Network - Design Document

## 1. System Architecture Overview

This project implements a binary neural network for time-series forecasting, specifically predicting solar X-ray flux from GOES satellite data. The architecture follows classical backpropagation principles but with a critical modification: **all weights are quantized to binary values (+1 or -1)**.

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                    BITNET NETWORK                        │
│                                                         │
│  Input Layer (N=20 units)                               │
│       │                                                   │
│       ▼                                                   │
│  Hidden Layer (16 units)                                  │
│       • Binary weights (+1/-1)                            │
│       • Sigmoid activation                                │
│       • Momentum updates                                  │
│       │                                                   │
│       ▼                                                   │
│  Output Layer (M=1 unit)                                  │
│       • Binary weights (+1/-1)                            │
│       • Sigmoid activation                                │
└─────────────────────────────────────────────────────────┘
```

## 2. Core Design Decisions

### 2.1 Binary Weight Quantization

**Decision**: Map all floating-point weights to binary values during the forward pass.

**Rationale**: 
- Reduces memory footprint by 32x compared to FP32
- Enables bitwise XOR-based dot product computation
- Inspired by BitNet's approach to efficient LLM inference

**Implementation**:
```c
// Sign-based quantization: +1 if positive, -1 if negative
Net->Layer[l]->BWeight[i][j] = (Net->Layer[l]->Weight[i][j] >= 0) ? 1 : -1;
```

### 2.2 Dual-Weight Representation

**Decision**: Maintain both floating-point weights and binary weights simultaneously.

**Rationale**:
- Floating-point weights store gradient information for updates
- Binary weights are used during inference (forward pass)
- Enables continuous learning while maintaining binary inference efficiency

### 2.3 Training Strategy

**Decision**: Use momentum-based SGD with early stopping.

**Parameters**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning rate (η) | 0.25 | Step size for weight updates |
| Momentum (α) | 0.9 | Smoothing factor for gradient history |
| Gain | 1.0 | Sigmoid steepness |
| Stop threshold | 1.2 × min_test_error | Early stopping condition |

**Training Loop**:
```
1. Randomly sample training example
2. Forward pass (using binary weights)
3. Compute error at output layer
4. Backpropagate error (using binary weights)
5. Update floating-point weights with momentum
6. Re-quantize all weights to binary
7. Save best weights if test error improves
```

### 2.4 Network Topology

**Decision**: 3-layer feedforward network: Input → Hidden → Output

| Layer | Units | Connections |
|-------|-------|-------------|
| Input | N = 20 | Sliding window of past flux values |
| Hidden | 16 | Fully connected |
| Output | M = 1 | Single regression value |

**Justification**: The 20-unit input layer captures a 20-year sliding window of solar flux history, providing sufficient temporal context for prediction.

## 3. Data Flow Design

### 3.1 Input Processing

```c
void SetInput(NET* Net, REAL* Input) {
    // Map input array to input layer outputs
    for (i=1; i<=Net->InputLayer->Units; i++) {
        Net->InputLayer->Output[i] = Input[i-1];
    }
}
```

### 3.2 Normalization

**Decision**: Min-max normalization to [LO, HI] = [0.1, 0.9].

```c
SolarFlux_[Year] = ((SolarFlux[Year]-Min) / (Max-Min)) * (HI-LO) + LO;
```

### 3.3 Forward Propagation

**Key insight**: Use binary weights for the dot product computation:

```c
void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper) {
    for (i=1; i<=Upper->Units; i++) {
        Sum = 0;
        for (j=0; j<=Lower->Units; j++) {
            // Binary weight multiplication
            Sum += Upper->BWeight[i][j] * Lower->Output[j];
        }
        // Sigmoid activation
        Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
    }
}
```

### 3.4 Backpropagation

**Design choice**: Use binary weights in the error propagation equation:

```c
void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower) {
    for (i=1; i<=Lower->Units; i++) {
        Err = 0;
        for (j=1; j<=Upper->Units; j++) {
            // Binary weight in backpropagation
            Err += Upper->BWeight[j][i] * Upper->Error[j];
        }
        Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
    }
}
```

### 3.5 Weight Update with Quantization

**Critical design point**: Weights are updated as floats but re-quantized after each pass:

```c
void AdjustWeights(NET* Net) {
    for (l=1; l<NUM_LAYERS; l++) {
        for (i=1; i<=Net->Layer[l]->Units; i++) {
            for (j=0; j<=Net->Layer[l-1]->Units; j++) {
                // Momentum-based update
                Net->Layer[l]->Weight[i][j] += 
                    Net->Eta * Err * Out + Net->Alpha * dWeight;
                
                // Store new gradient direction
                Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
            }
        }
    }
    // Re-quantize after update
    BinaryQuantizeWeights(Net);
}
```

## 4. Evaluation Design

### 4.1 Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| NMSE | Σ(E²) / Σ(Mean - y)² | Normalized against mean predictor baseline |
| Open-loop error | Predict using actual past values | Pure model accuracy |
| Closed-loop error | Predict using own predictions recursively | Long-term stability |

### 4.2 Evaluation Modes

**Open-Loop Prediction**: Each prediction uses the true historical values as input. Measures raw model accuracy without error accumulation.

**Closed-Loop Prediction**: Each prediction uses the model's previous output as input, simulating real-world deployment where ground truth is unavailable.

```c
void EvaluateNet(NET* Net) {
    for (Year=EVAL_LWB; Year<=EVAL_UPB; Year++) {
        // Open-loop: use actual data
        SimulateNet(Net, &(SolarFlux [Year-N]), Output,  &(SolarFlux [Year]), FALSE);
        
        // Closed-loop: use previous prediction
        SimulateNet(Net, &(SolarFlux_[Year-N]), Output_, &(SolarFlux_[Year]), FALSE);
        SolarFlux_[Year] = Output_[0];  // Feed back prediction
    }
}
```

## 5. Memory Layout Design

### LAYER Structure

```c
typedef struct {
    INT           Units;      // Number of neurons
    REAL*         Output;     // Activation values [Units+1] (index 0 = bias)
    REAL*         Error;      // Backpropagated error [Units+1]
    REAL**        Weight;     // Float weights for update [Units][Input+1]
    REAL**        WeightSave; // Saved best weights [Units][Input+1]
    REAL**        dWeight;    // Previous gradient [Units][Input+1]
    INT**         BWeight;    // Binary weights for inference [Units][Input+1]
} LAYER;
```

### Memory Considerations

- **Binary weights**: Use `INT` (4 bytes) instead of optimized bit-packing due to C implementation simplicity
- **Bias handling**: Index 0 in each layer is reserved as bias term
- **Weight saving**: Full float copy saved when test error improves for restoration

## 6. Training Control Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                             │
│                                                              │
│  Initialize: weights, random seed, data normalization        │
│                                                              │
│  Loop:                                                       │
│    ├─ TrainNet(10 epochs)                                    │
│    │   └─ Random sample + SimulateNet (training mode)        │
│    │                                                        │
│    ├─ TestNet()                                             │
│    │   ├─ Evaluate on training set                          │
│    │   └─ Evaluate on test set                              │
│    │                                                        │
│    ├─ If TestError < MinTestError:                          │
│    │   └─ SaveWeights()                                     │
│    │                                                      │
│    └─ If TestError > 1.2 × MinTestError:                   │
│        └─ RestoreWeights(), Stop = TRUE                     │
│                                                              │
│  Final: TestNet() + EvaluateNet()                           │
└──────────────────────────────────────────────────────────────┘
```

## 7. Key Design Trade-offs

| Trade-off | Decision | Rationale |
|-----------|----------|-----------|
| Binary inference vs. float training | Float weights updated, binary used for inference | Preserves gradient information while enabling efficient inference |
| Sign-based quantization | ±1 mapping | Simplest form of 1-bit representation |
| Post-update re-quantization | Quantize after each weight update | Maintains binary constraint during learning |
| Early stopping threshold | 20% degradation | Prevents overfitting while allowing exploration |
