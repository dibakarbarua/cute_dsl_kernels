# CuTe DSL Kernels
CuTe DSL Kernels replicating NVIDIA's CUTLASS implementations.

## Kernel 00: Blackwell FlashAttention4
- CUTLASS Example: [Example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha)
- Roofline: (1.4 PFlops @ bfloat16)
- FlashAttention4 Description: (Slide 55) [TriDao @ LLMSys](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-20-FlashAttention_tridao-cac5b634b4ad77cb027451422b07ae75.pdf)
- Kernel Parameters
   - Fixed Length Sequence
   - No masking
   - Same Sequence Length across batches
- Useful examples from NVIDIA using CuTeDSL:
   - [Dense GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm.py)
   - [FMHA](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py)
- Tutorial Notebooks:
   - [NVIDIA Notebooks](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks)
