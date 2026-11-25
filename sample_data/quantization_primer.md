# Quantization Techniques for Tiny Models

Quantization reduces the precision of model weights from 16-bit floating point (FP16) to 4-bit or 8-bit integers (INT4 / INT8).
This allows large models to fit into smaller memory footprints:
- **AWQ (Activation-aware Weight Quantization)**: Preserves salient weights to maintain high reasoning quality.
- **GGUF Format**: Used by llama.cpp and Ollama for fast, single-file CPU and Metal GPU execution on Apple Silicon.
- **Memory Footprint**: A 3B parameter model quantized to 4-bit consumes only ~2.0 GB of RAM, making it ideal for edge computing.
