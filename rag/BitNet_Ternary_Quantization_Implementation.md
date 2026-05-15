keywords: bitnet, ternary, quantization, weights, linux

# BitNet Ternary Quantization Implementation

## Overview
BitNet uses ternary weights (-1, 0, +1) for efficient inference. This is 1.58-bit quantization (log2(3) ≈ 1.58).

## Key Concepts

### 1. Weight Binarization (Sign Function)
```
W_b = Sign(W) = +1 if W >= 0, -1 if W < 0
```

### 2. Scaling Factor
```
alpha = (1/nm) * sum(|W_ij|)  // mean of absolute values
```

### 3. Absmean Quantization
The b1.58 paper introduced absmean quantization for matching FP16 performance.

### 4. I2_S Format (from bitnet.cpp)
- Packs 4 rows of weights into 1 byte each
- Each weight encoded as 2 bits: 00=-1, 01=0, 10=+1
- Scale factor stored separately

## Implementation Pattern (Linux C Style)

```c
// Weight packing for I2_S format
void pack_weights_i2s(const float* weights, uint8_t* packed, int n) {
    // Convert to ternary: -1, 0, +1
    for (int i = 0; i < n; i++) {
        if (weights[i] > 0) ternary[i] = 2;   // +1
        else if (weights[i] < 0) ternary[i] = 0;  // -1
        else ternary[i] = 1;  // 0
    }
    
    // Pack 4 weights into 1 byte
    for (int i = 0; i < n/4; i++) {
        packed[i] = (ternary[i*4] << 6) | 
                    (ternary[i*4+1] << 4) | 
                    (ternary[i*4+2] << 2) | 
                    (ternary[i*4+3] << 0);
    }
}

// Matrix multiplication using ternary weights
// Instead of: C = A * W (multiplications)
// Use: C = sum(sign(A) * sign(W)) (additions/subtractions)
```

## Key Optimizations from bitnet.cpp
1. Parallel weight/activation computation
2. Configurable tiling block sizes
3. Embedding quantization (Q6_K format)
4. Native I2_S GEMM/GEMV support

## References
- BitNet v1: https://arxiv.org/abs/2310.16792
- BitNet b1.58: https://arxiv.org/pdf/2411.05882
- bitnet.cpp: https://github.com/microsoft/BitNet
