# Apex

Apex is a PyTorch extension maintained by NVIDIA that provides utilities for mixed precision training and distributed training to accelerate deep learning workloads on NVIDIA GPUs.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [NVIDIA PyTorch Containers](#nvidia-pytorch-containers)
  - [From Source](#from-source)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

Apex provides several key capabilities for high-performance deep learning:

**Fused Optimizers** such as FusedAdam, FusedSGD, and FusedLAMB batch parameter updates into single kernel launches for improved performance.

**Fused Normalization Layers** including FusedLayerNorm and FusedRMSNorm combine multiple operations into optimized CUDA kernels.

**Distributed Training Optimizers** like DistributedFusedAdam implement ZeRO-2 memory optimization, sharding optimizer state across GPUs to enable training of larger models.

**Transformer Parallelism** utilities support tensor model parallelism, pipeline model parallelism, and sequence parallelism for scaling transformer models across multiple GPUs.

## Requirements

Apex requires the following:

- Python 3.8 or later
- PyTorch 1.0 or later (nightly recommended for latest features)
- NVIDIA GPU with CUDA support (for CUDA extensions)
- CUDA Toolkit matching your PyTorch CUDA version
- Ninja build system (recommended for faster compilation)

Specific extensions have additional requirements:

| Extension | Requirement |
|-----------|-------------|
| `fused_weight_gradient_mlp_cuda` | CUDA >= 11.0 |
| `fmhalib` (fused multi-head attention) | CUDA >= 11.0, SM 80/90 (Ampere/Hopper GPUs) |
| `cudnn_gbn_lib` | cuDNN >= 8.5 |
| `fused_conv_bias_relu` | cuDNN >= 8.4 |
| `nccl_p2p_cuda` | NCCL >= 2.10 |
| `_apex_nccl_allocator` | NCCL >= 2.19 |

## Installation

### NVIDIA PyTorch Containers

The easiest way to use Apex is through NVIDIA PyTorch containers available on NGC, which come with all extensions pre-built:

```bash
docker pull nvcr.io/nvidia/pytorch:24.01-py3
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
```

See the [NGC PyTorch container page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) and [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) for more details.

### From Source

Clone the repository and install with the desired extensions:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
```

**Recommended installation with CUDA extensions:**

```bash
# Install with core CUDA extensions
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# For faster compilation, enable parallel building
NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
```

**Install with additional extensions:**

```bash
# Include fused multi-head attention and conv-bias-relu
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FMHA=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation .

# Install all contrib extensions
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_ALL_CONTRIB_EXT=1 pip install -v --no-build-isolation .
```

**Python-only installation (no CUDA extensions):**

```bash
pip install -v --no-build-isolation .
```

This omits fused kernels for optimizers, normalization layers, and other performance optimizations. The Python-only build is useful for development or environments without CUDA.

### Available Extensions

| Module | Environment Variable | Description |
|--------|---------------------|-------------|
| `apex_C` | `APEX_CPP_EXT=1` | Core C++ extensions |
| `amp_C`, `syncbn`, `fused_layer_norm_cuda`, `mlp_cuda` | `APEX_CUDA_EXT=1` | Core CUDA extensions |
| `distributed_adam_cuda` | `APEX_DISTRIBUTED_ADAM=1` | ZeRO-2 Adam optimizer |
| `distributed_lamb_cuda` | `APEX_DISTRIBUTED_LAMB=1` | ZeRO-2 LAMB optimizer |
| `fmhalib` | `APEX_FMHA=1` | Fused multi-head attention |
| `fast_multihead_attn` | `APEX_FAST_MULTIHEAD_ATTN=1` | Fast multi-head attention |
| `fast_layer_norm` | `APEX_FAST_LAYER_NORM=1` | Alternative LayerNorm implementation |
| `fused_conv_bias_relu` | `APEX_FUSED_CONV_BIAS_RELU=1` | Fused convolution operations |

See the full list of extensions in the [Custom Extensions](#custom-cuda-extensions) section below.

## Quick Start

### Fused Optimizers

Replace standard PyTorch optimizers with fused versions for better performance:

```python
from apex.optimizers import FusedAdam

# Instead of: optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = FusedAdam(model.parameters(), lr=1e-3)
```

### Fused Layer Normalization

Use fused normalization layers for faster training:

```python
from apex.normalization import FusedLayerNorm, FusedRMSNorm

# Instead of: norm = torch.nn.LayerNorm(hidden_size)
norm = FusedLayerNorm(hidden_size)

# For RMSNorm (common in LLMs)
rms_norm = FusedRMSNorm(hidden_size)
```

### Distributed Training with ZeRO-2

Use DistributedFusedAdam to shard optimizer state across GPUs:

```python
from apex.contrib.optimizers import DistributedFusedAdam

optimizer = DistributedFusedAdam(
    model.parameters(),
    lr=1e-3,
    overlap_grad_sync=True,  # Overlap gradient sync with backward pass
)
```

### Mixed Precision Training Example

For a complete example of mixed precision training, see the [ImageNet training example](./examples/imagenet/):

```bash
cd examples/imagenet
python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 ./
```

## Project Structure

```
apex/
├── apex/                    # Python source code
│   ├── optimizers/          # Fused optimizers (FusedAdam, FusedSGD, etc.)
│   ├── normalization/       # Fused normalization layers
│   ├── transformer/         # Transformer parallelism utilities
│   └── contrib/             # Additional/experimental modules
│       ├── optimizers/      # Distributed optimizers (ZeRO-2)
│       ├── layer_norm/      # Alternative LayerNorm implementations
│       └── ...
├── csrc/                    # C++/CUDA source code
├── examples/                # Usage examples
│   ├── imagenet/            # ImageNet training with mixed precision
│   └── ...
├── tests/                   # Test suite
└── docs/                    # Documentation source
```

## Custom CUDA Extensions

The following table lists all available CUDA extensions and their build options:

| Module Name | Environment Variable | Install Option | Notes |
|-------------|---------------------|----------------|-------|
| `apex_C` | `APEX_CPP_EXT=1` | `--cpp_ext` | |
| `amp_C` | `APEX_CUDA_EXT=1` | `--cuda_ext` | |
| `syncbn` | `APEX_CUDA_EXT=1` | `--cuda_ext` | |
| `fused_layer_norm_cuda` | `APEX_CUDA_EXT=1` | `--cuda_ext` | [apex.normalization](./apex/normalization) |
| `mlp_cuda` | `APEX_CUDA_EXT=1` | `--cuda_ext` | |
| `scaled_upper_triang_masked_softmax_cuda` | `APEX_CUDA_EXT=1` | `--cuda_ext` | |
| `generic_scaled_masked_softmax_cuda` | `APEX_CUDA_EXT=1` | `--cuda_ext` | |
| `scaled_masked_softmax_cuda` | `APEX_CUDA_EXT=1` | `--cuda_ext` | |
| `fused_weight_gradient_mlp_cuda` | `APEX_CUDA_EXT=1` | `--cuda_ext` | Requires CUDA>=11 |
| `permutation_search_cuda` | `APEX_PERMUTATION_SEARCH=1` | `--permutation_search` | [apex.contrib.sparsity](./apex/contrib/sparsity) |
| `bnp` | `APEX_BNP=1` | `--bnp` | [apex.contrib.groupbn](./apex/contrib/groupbn) |
| `xentropy` | `APEX_XENTROPY=1` | `--xentropy` | [apex.contrib.xentropy](./apex/contrib/xentropy) |
| `focal_loss_cuda` | `APEX_FOCAL_LOSS=1` | `--focal_loss` | [apex.contrib.focal_loss](./apex/contrib/focal_loss) |
| `fused_index_mul_2d` | `APEX_INDEX_MUL_2D=1` | `--index_mul_2d` | [apex.contrib.index_mul_2d](./apex/contrib/index_mul_2d) |
| `fused_adam_cuda` | `APEX_DEPRECATED_FUSED_ADAM=1` | `--deprecated_fused_adam` | [apex.contrib.optimizers](./apex/contrib/optimizers) |
| `fused_lamb_cuda` | `APEX_DEPRECATED_FUSED_LAMB=1` | `--deprecated_fused_lamb` | [apex.contrib.optimizers](./apex/contrib/optimizers) |
| `fast_layer_norm` | `APEX_FAST_LAYER_NORM=1` | `--fast_layer_norm` | [apex.contrib.layer_norm](./apex/contrib/layer_norm) |
| `fmhalib` | `APEX_FMHA=1` | `--fmha` | [apex.contrib.fmha](./apex/contrib/fmha) |
| `fast_multihead_attn` | `APEX_FAST_MULTIHEAD_ATTN=1` | `--fast_multihead_attn` | [apex.contrib.multihead_attn](./apex/contrib/multihead_attn) |
| `transducer_joint_cuda` | `APEX_TRANSDUCER=1` | `--transducer` | [apex.contrib.transducer](./apex/contrib/transducer) |
| `transducer_loss_cuda` | `APEX_TRANSDUCER=1` | `--transducer` | [apex.contrib.transducer](./apex/contrib/transducer) |
| `cudnn_gbn_lib` | `APEX_CUDNN_GBN=1` | `--cudnn_gbn` | Requires cuDNN>=8.5, [apex.contrib.cudnn_gbn](./apex/contrib/cudnn_gbn) |
| `peer_memory_cuda` | `APEX_PEER_MEMORY=1` | `--peer_memory` | [apex.contrib.peer_memory](./apex/contrib/peer_memory) |
| `nccl_p2p_cuda` | `APEX_NCCL_P2P=1` | `--nccl_p2p` | Requires NCCL >= 2.10, [apex.contrib.nccl_p2p](./apex/contrib/nccl_p2p) |
| `fast_bottleneck` | `APEX_FAST_BOTTLENECK=1` | `--fast_bottleneck` | Requires `peer_memory_cuda` and `nccl_p2p_cuda`, [apex.contrib.bottleneck](./apex/contrib/bottleneck) |
| `fused_conv_bias_relu` | `APEX_FUSED_CONV_BIAS_RELU=1` | `--fused_conv_bias_relu` | Requires cuDNN>=8.4, [apex.contrib.conv_bias_relu](./apex/contrib/conv_bias_relu) |
| `distributed_adam_cuda` | `APEX_DISTRIBUTED_ADAM=1` | `--distributed_adam` | [apex.contrib.optimizers](./apex/contrib/optimizers) |
| `distributed_lamb_cuda` | `APEX_DISTRIBUTED_LAMB=1` | `--distributed_lamb` | [apex.contrib.optimizers](./apex/contrib/optimizers) |
| `_apex_nccl_allocator` | `APEX_NCCL_ALLOCATOR=1` | `--nccl_allocator` | Requires NCCL >= 2.19, [apex.contrib.nccl_allocator](./apex/contrib/nccl_allocator) |
| `_apex_gpu_direct_storage` | `APEX_GPU_DIRECT_STORAGE=1` | `--gpu_direct_storage` | [apex.contrib.gpu_direct_storage](./apex/contrib/gpu_direct_storage) |

Build all contrib extensions at once by setting `APEX_ALL_CONTRIB_EXT=1`.

## Documentation

- [API Documentation](https://nvidia.github.io/apex/) - Full API reference for Apex modules
- [Fused Optimizers](https://nvidia.github.io/apex/optimizers.html) - Documentation for FusedAdam, FusedSGD, and other optimizers
- [Fused Layer Norm](https://nvidia.github.io/apex/layernorm.html) - Documentation for FusedLayerNorm and FusedRMSNorm
- [ImageNet Example](./examples/imagenet/) - Complete example of mixed precision training
- [Mixed Precision References](https://github.com/mcarilli/mixed_precision_references) - GTC and PyTorch DevCon slides on mixed precision training

## Contributing

Contributions to Apex are welcome. When contributing, please:

1. Fork the repository and create a feature branch
2. Ensure your code follows the existing style conventions
3. Add tests for new functionality where applicable
4. Submit a pull request with a clear description of your changes

For bug reports and feature requests, please open an issue on the [GitHub repository](https://github.com/NVIDIA/apex/issues).

## License

Apex is released under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file for details.

---

*Originally written and maintained by contributors and [Devin](https://app.devin.ai), with updates from the core team.*
