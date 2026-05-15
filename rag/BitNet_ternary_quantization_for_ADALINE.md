keywords: BitNet, ternary, quantization, ADALINE

BitNet b1.58 uses ternary weights {-1, 0, +1} with absmean scaling: scale = mean(|weights|), then w_quant = +1 if w >= scale, -1 if w <= -scale, else 0. This preserves gradient flow during training and enables multiplication-free inference via sign flips and additions. Applied to ADALINE: replace REAL weights with INT {-1,0,1}, use absmean scaling for quantization, eliminate floating-point multiplies in dot product by using conditional adds/subtracts.
