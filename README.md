# cute_dsl_kernels
CuTe DSL Kernels replicating NVIDIA's CUTLASS implementations.

## Kernel 00: Blackwell FMHA
- CUTLASS Example: [text](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha)
- Roofline: (1.4 PFlops @ bfloat16)
- FlashAttention4 Description: (Slide 55) [text](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-20-FlashAttention_tridao-cac5b634b4ad77cb027451422b07ae75.pdf)
- Kernel Parameters
   - Fixed Length Sequence
   - No masking
   - Same Sequence Length across batches
