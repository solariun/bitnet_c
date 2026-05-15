keywords: BitNet, quantization, ternary, machine-learning, skill

# BitNet b1.58 Quantization (Ternary Weights)

## Core Concept
BitNet b1.58 uses ternary weights $\{-1, 0, +1\}$, which provides approximately 1.58 bits of information per weight. This allows for massive compression and efficient integer-based matrix multiplication.

## Quantization Mechanism (Absmean)
To map high-precision (FP16/BF16) weights to ternary values:
1. **Calculate Scale**: $\text{scale} = \text{mean}(|W_{fp}|)$
2. **Quantize**: $W_q = \text{round}\left(\frac{W_{fp}}{\text{scale}}\right)$
3. **Clamp**: Ensure $W_q \in \{-1, 0, 1\}$

## Inference (Forward Pass)
The computation is performed using the quantized weights and then rescaled:
$$\text{Output} = \text{scale} \times \sum (W_q \cdot \text{Input})$$
This allows the core operation to be an integer multiply-accumulate (MAC), followed by a single floating-point multiplication per layer/channel.

## Training (Backward Pass)
To train quantized models, the **Straight-Through Estimator (STE)** is used. 
1. Maintain high-precision "master weights" ($W_{fp}$).
2. During the forward pass, use $W_q$.
3. During the backward pass, calculate gradients based on the error and apply them directly to the master weights:
   $$W_{fp} \leftarrow W_{fp} + \eta \cdot \text{error} \cdot \text{input}$$
This allows the model to learn subtle weight adjustments that eventually trigger a change in the quantized ternary state.

## Key Benefits
- **Compression**: ~10x reduction in memory compared to FP16.
- **Efficiency**: Replaces expensive floating-point multiplications with integer additions/subtractions (since weights are $\{-1, 0, 1\}$).
- **Scalability**: Mitigates quantization error compounding through learned scaling factors and normalization.
