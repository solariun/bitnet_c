# BitNet - Reference Document

## Background: BitNet Architecture from Original Research

This document provides comprehensive reference material on the BitNet architecture, based on the original academic papers and official documentation.

---

## 1. The BitNet Paper

### Citation

**Title**: BitNet: Scaling 1-bit Transformers for Large Language Models

**Authors**: Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei

**Published**: October 17, 2023

**arXiv**: [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)

**Venue**: Computation and Language (cs.CL)

### Abstract Summary

The BitNet paper addresses the growing challenges of deploying large language models, including high energy consumption and resource requirements. The authors introduce **BitNet**, a scalable and stable 1-bit Transformer architecture designed specifically for LLMs.

Key contributions include:
- **BitLinear**: A drop-in replacement for `nn.Linear` layers that enables training 1-bit weights from scratch
- Competitive language modeling performance compared to state-of-the-art 8-bit quantization methods and FP16 baselines
- Substantial reductions in memory footprint and energy consumption
- Demonstration of a scaling law similar to full-precision Transformers, suggesting potential for effective scaling to larger models

---

## 2. BitNet Architecture Fundamentals

### 2.1 Core Philosophy

BitNet represents an extreme form of model quantization where weights are constrained to minimal precision, enabling:
- Drastically reduced memory requirements for weight storage
- Highly efficient bitwise computations during inference
- Lower energy consumption and deployment costs

### 2.2 BitLinear Layer

The **BitLinear layer** is the foundational innovation of BitNet, replacing standard full-precision linear layers (`torch.nn.Linear`).

#### Weight Quantization

BitNet employs two primary quantization schemes:

| Scheme | Values | Bits per Parameter | Description |
|--------|--------|-------------------|-------------|
| **b1 (Binary)** | {-1, +1} | 1.0 | Strict binary representation |
| **b1.58 (Ternary)** | {-1, 0, +1} | ~1.58 | Ternary representation (log₂(3) ≈ 1.585 bits) |

#### Quantization Process

**Absmean Quantization** (for b1):
```
w_binary = sign(w_float)
where:
  +1 if w_float ≥ 0
  -1 if w_float < 0
```

**Absmax Ternary Quantization** (for b1.58):
- Maps weights to ternary values {-1, 0, +1}
- Uses scaling techniques and shadow weights
- Employs straight-through estimator for gradient flow during training

#### Activation Quantization

Activations flowing through BitLinear layers are quantized to **8-bit integers** using absmax quantization strategy, applied per-token.

### 2.3 Normalization

BitNet incorporates **subln normalization** (Wang et al., 2022) to enhance training stability, particularly beneficial in quantized training regimes.

### 2.4 Feed-Forward Network Modifications

Unlike standard Transformers that use SwiGLU activation, BitNet b1.58 employs **squared ReLU** (`ReLU²`):

```
FFN(x) = W₂ · ReLU²(W₁ · x + b₁) + b₂
```

This choice is motivated by its potential to improve model sparsity and computational characteristics within the 1-bit context.

---

## 3. BitNet b1.58 2B4T Technical Report

### Citation

**Title**: BitNet b1.58 2B4T Technical Report

**Authors**: Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu, Ting Song, Yan Xia, Furu Wei

**Published**: April 2025

**arXiv**: [arXiv:2504.12285](https://arxiv.org/html/2504.12285v2)

### Model Overview

BitNet b1.58 2B4T is the **first open-source, native 1-bit Large Language Model** at the 2-billion parameter scale.

#### Training Details

| Aspect | Specification |
|--------|---------------|
| Parameters | 2 billion |
| Pre-training tokens | 4 trillion |
| Weight precision | 1.58 bits (ternary {-1, 0, +1}) |
| Activation precision | 8-bit integers |
| Format notation | W1.58A8 |

#### Training Pipeline

1. **Pre-training**: 4T tokens with custom learning rate and weight decay schedules
2. **Supervised Fine-tuning (SFT)**: Instruction-following capabilities
3. **Direct Preference Optimization (DPO)**: Alignment for conversational ability

### Performance Claims

BitNet b1.58 2B4T demonstrates performance on par with leading open-weight, full-precision LLMs of similar size across:
- Language understanding
- Mathematical reasoning
- Coding proficiency
- Conversational ability

---

## 4. bitnet.cpp Inference Framework

### Citation

**Repository**: [microsoft/BitNet](https://github.com/microsoft/BitNet)

**Description**: Official inference framework for 1-bit LLMs with optimized kernels for fast and lossless inference.

### Performance Benchmarks

#### ARM CPU Performance
| Metric | Range |
|--------|-------|
| Speedup | 1.37x – 5.07x |
| Energy reduction | 55.4% – 70.0% |

#### x86 CPU Performance
| Metric | Range |
|--------|-------|
| Speedup | 2.37x – 6.17x |
| Energy reduction | 71.9% – 82.2% |

#### Large Model Inference
- **100B BitNet b1.58 model**: Runs on a single CPU at ~5-7 tokens per second (comparable to human reading speed)

### Key Features

- Optimized kernels for CPU and GPU architectures
- Parallel kernel implementations with configurable tiling
- Embedding quantization support
- NPU support planned

---

## 5. Weight Packing Format

### Ternary Weight Storage

The ternary weights {-1, 0, +1} are packed efficiently:

| Value | Binary Encoding |
|-------|-----------------|
| -1 | `10` (or `01` depending on convention) |
| 0 | `00` |
| +1 | `11` |

This allows **2 ternary values per byte**, achieving effective storage of ~1.58 bits per parameter.

### Scale Factor Organization

Weights are organized in blocks with associated scale factors for dequantization during inference:
- Block-based organization enables efficient memory access patterns
- Scale factors allow lossless reconstruction of ternary values

---

## 6. Comparison with Traditional Quantization

| Aspect | Post-Training Quantization (PTQ) | BitNet Native 1-bit |
|--------|----------------------------------|---------------------|
| Training approach | Applied to pre-trained FP model | Trained from scratch |
| Performance impact | Can cause degradation | Near-parity maintained |
| Memory efficiency | Moderate improvement | Maximum reduction |
| Computational efficiency | Limited benefit | Significant bitwise optimization |

---

## 7. Key Architectural Components Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    BitNet Architecture                        │
│                                                             │
│  ┌───────────┐    ┌──────────────────┐    ┌───────────┐     │
│  │   Input    │───▶│   BitLinear      │───▶│ Attention │     │
│  │  Tokens    │    │  (W1.58→A8)      │    │  (AFT)    │     │
│  └───────────┘    └──────────────────┘    └───────────┘     │
│                           │                    │              │
│                           ▼                    ▼              │
│                    ┌──────────────────┐    ┌───────────┐     │
│                    │   BitLinear      │◀──▶│  subln    │     │
│                    │  (W1.58→A8)      │    │ Normaliz. │     │
│                    └──────────────────┘    └───────────┘     │
│                           │                                    │
│                           ▼                                    │
│                    ┌──────────────────┐                       │
│                    │  FFN (ReLU²)     │                       │
│                    │  BitLinear       │                       │
│                    └──────────────────┘                       │
│                                                             │
│  Key Innovations:                                           │
│  • BitLinear: Binary/ternary weights, 8-bit activations     │
│  • AFT Attention: Alternative to standard QKV attention      │
│  • subln: Sub-layer normalization for stability             │
│  • ReLU²: Squared ReLU in FFN for sparsity                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Additional References

### Related Publications

1. **Wang et al., 2022**: "subln" normalization technique for Transformer stability
2. **Shazeer, 2020**: SwiGLU activation function (standard alternative to ReLU²)
3. **Vaswani et al., 2017**: Original Transformer architecture

### Model Availability

| Model | Hugging Face ID | Description |
|-------|-----------------|-------------|
| BitNet b1.58 2B4T (1.58-bit) | `bitnet-b1.58-2B-4T` | Inference weights |
| BitNet b1.58 2B4T (bf16) | `bitnet-b1.58-2B-4T-bf16` | Training master weights |
| BitNet b1.58 2B4T (gguf) | `bitnet-b1.58-2B-4T-gguf` | GGUF format for bitnet.cpp |

### Official Resources

- **Technical Report**: https://arxiv.org/html/2504.12285v2
- **GitHub Repository**: https://github.com/microsoft/BitNet
- **Demo**: https://aka.ms/bitnet-demo
- **Optimization Guide**: Available in repository docs

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **BitLinear** | Custom linear layer with ternary weights and 8-bit activations |
| **W1.58A8** | Weight precision (1.58-bit) and activation precision (8-bit) notation |
| **Absmean** | Quantization method using mean of absolute values for scaling |
| **Absmax** | Quantization method using maximum of absolute values for scaling |
| **AFT** | Attention with Function Transformer (alternative attention mechanism) |
| **subln** | Sub-layer normalization for training stability |
| **NMSE** | Normalized Mean Squared Error - evaluation metric |
| **PTQ** | Post-Training Quantization |
| **DPO** | Direct Preference Optimization |