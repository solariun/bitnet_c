keywords: Adaline, BitNet, ternary, quantization, STE, C, neural-network

# BitNet-Adaline: Ternary Quantization for Adaline Networks

## Concept
Rewrite an Adaline (adaptive linear neuron) network to use BitNet-style ternary {-1, 0, +1} weights while preserving the exact same forward/backward computation semantics.

## Key Design Decisions
1. **Ternary weight initialization**: Use absmean quantization — generate random FP weights, compute scale = mean(|W|), then W_q = sign(W_fp) with clamping to {-1, 0, +1} based on |W_fp/scale|.
2. **Forward pass**: output = scale * sum(W_q[i] * input[i]) — the scale factor preserves magnitude information lost by ternary quantization.
3. **Training**: Straight-Through Estimator (STE). Maintain high-precision master weights for gradient updates. During forward pass, use quantized ternary weights. Gradients flow through STE to update master weights directly.
4. **Weight update rule unchanged**: W_master += eta * error * input (same delta rule as original Adaline).
5. **Quantization applied at each step**: After updating master weights, re-quantize to ternary for the next forward pass.

## Quantization Formula
- scale = mean(|W_master|) over all weights
- W_q[i] = +1 if W_master[i] > scale * 0.5
- W_q[i] = -1 if W_master[i] < -scale * 0.5  
- W_q[i] = 0 otherwise (|W_master[i]| <= scale * 0.5)

## Forward Pass
- activation[i] = scale * sum_j(W_q[i][j] * input[j])
- output[i] = sign(activation[i]) → +1 or -1

## Benefits over original
- Weights stored as int8_t (ternary) instead of double — ~8x memory reduction
- Multiplications replaced by additions/subtractions (W_q is only -1, 0, +1)
- Same classification behavior preserved via scale factor and STE training
