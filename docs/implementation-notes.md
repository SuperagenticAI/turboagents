# Implementation Notes

- The initial implementation prioritizes a correct, inspectable scaffold over premature optimization.
- The current `quant.pipeline` now includes:
  - Fast Walsh-Hadamard rotation with cached sign patterns
  - structured PolarQuant-style angle/radius encoding
  - seeded QJL-style Gaussian sign sketch for residual reconstruction
- The implementation is still an approximation of the paper's final production path, especially around:
  - final Lloyd-Max tables
  - large-scale benchmark reproduction
  - native engine kernels
- This Mac should be treated as a low-RAM development target:
  - use synthetic benchmarks and adapter dry-runs here
  - use the higher-memory Mac for large-model and long-context evaluation
