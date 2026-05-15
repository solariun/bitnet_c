# Adaline Network with BitNet Quantization

This project is a modernized version of the classic Adaline neural network implementation from Karsten Kutza, updated to follow Linux coding standards and incorporating concepts from Microsoft's BitNet quantization research.

## Original Implementation

The original ADELIDE.c was a pattern recognition system that classifies handwritten digits (0-9) using an Adaline neural network. It was originally written in the 1990s and used traditional C programming practices.

## Modernizations

### Linux Coding Standards
- Updated to C99 standard
- Proper includes and header organization
- Added `stdbool.h` for boolean types
- Improved memory management with proper error checking
- Better function signatures and parameter handling
- Removed deprecated constructs

### BitNet Quantization Concepts
- Implemented ternary quantization (similar to BitNet's approach)
- Added weight quantization during training
- Used bitnet-inspired quantization functions
- Applied quantization-aware training concepts

## Features

1. **Pattern Recognition**: Classifies 10 handwritten digit patterns (0-9)
2. **Adaline Network**: Uses the classic Adaline learning algorithm
3. **Quantization**: Implements ternary quantization similar to BitNet's approach
4. **Logging**: Outputs recognition results to ADALINE.txt

## Building

```bash
make clean
make all
```

## Running

```bash
make run
```

The program will output training progress and save the final recognition results to `ADALINE.txt`.

## Files

- `adelide_modern.c`: The modernized source code
- `Makefile`: Build instructions
- `README.md`: This documentation

## Output Format

The output file `ADALINE.txt` contains:
- Input patterns (5x7 grid of 'O' and spaces)
- Recognition results (digit 0-9)

Example:
```
 OOO 
O   O
O   O
O   O
O   O
O   O
 OOO  -> 0
```

## References

- Original Adaline implementation by Karsten Kutza
- Microsoft BitNet research for 1-bit Transformers
- Quantization-Aware Training concepts