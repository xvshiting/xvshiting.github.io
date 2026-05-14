---
layout: post
author: willXu
tag: [Speculative Decoding, LLM, Deep Learning, NLP]
---

# Speculative Decoding: A Technical Survey

> **Date**: 2026-05-14
> **Status**: Draft

Speculative decoding is a promising inference acceleration technique for large language models (LLMs) that achieves **lossless speedup** without modifying the target model's output distribution. This survey covers the core principles, key algorithms, recent advances, and practical applications.

## 1. Background and Motivation

### The Inference Bottleneck

Autoregressive decoding requires a complete model forward pass for each token generation. This creates three critical problems:

- **Memory bandwidth bottleneck**: Billions of parameters must be loaded from HBM to GPU cache for each step, but the actual compute batch is tiny
- **Low compute utilization**: GPU parallel capacity is underutilized during single-token generation
- **High latency**: Decoding K tokens requires K sequential model calls

For example, LLaMA-70B typically achieves only 10-20 tokens/s — far below the ideal speed for interactive applications.

### Core Insight

> Language modeling tasks often contain easier sub-tasks that can be well-approximated by more efficient models.

The basic workflow:
1. **Draft**: A lightweight draft model quickly generates multiple candidate tokens
2. **Verify**: The target model verifies all candidates in a single forward pass
3. **Accept/Reject**: Correct tokens are accepted; incorrect ones are rejected and resampled

Advantages:
- ✅ **Lossless acceleration**: Output distribution is identical to the target model
- ✅ **No retraining required**: Directly applicable to existing models
- ✅ **Efficient parallelism**: Multiple serial calls merge into one parallel verification

---

## 2. Core Algorithms

### 2.1 Speculative Sampling (DeepMind, 2023)

**Paper**: "Accelerating Large Language Model Decoding with Speculative Sampling" ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318))

**Mechanism**:

1. Draft model generates γ candidate tokens: x₁, x₂, ..., xᵧ
2. Target model computes probability distributions for all positions in parallel
3. For each candidate xᵢ:
   - If q(xᵢ) ≥ p(xᵢ): accept directly (q: draft probability, p: target probability)
   - If q(xᵢ) < p(xᵢ): accept with probability min(1, p(xᵢ)/q(xᵢ))
4. If rejected, sample from the adjusted distribution: p'(x) = max(0, p(x) - q(x)) / Z

**Key property**: The modified rejection sampling scheme guarantees the output distribution matches the target model exactly (within hardware precision).

### 2.2 Speculative Decoding (Google, 2023)

**Paper**: "Fast Inference from Transformers via Speculative Decoding" ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192), ICML 2023 Oral)

**Contributions**:
- Proposed a speculative decoding framework applicable to T5 and similar models
- Achieved 2X-3X speedup on T5-XXL
- Key observation: parallel scoring has similar latency to single-token sampling

---

## 3. Advanced Techniques

### 3.1 Tree-based Speculative Decoding (SpecInfer)

**Paper**: "Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification" ([arXiv:2305.09781](https://arxiv.org/abs/2305.09781), ASPLOS'24)

Organizes candidate tokens as a **tree structure** rather than a linear sequence:

```
        [root]
       /  |  \
     x_1 x_2 x_3  (candidates from multiple draft models)
    /|   |   |\
  ...  ...  ...  (tree expansion)
```

**Advantages**: Multiple draft models provide diverse candidate paths; a single verification pass checks the entire tree.

**Performance**: Distributed inference 1.5-2.8x speedup; Offloading inference 2.6-3.5x speedup.

### 3.2 Feature-level Speculation (EAGLE)

**Paper**: "Speculative Sampling Requires Rethinking Feature Uncertainty" ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077))

> Feature-level (second-to-last layer) autoregression is simpler than token-level, but has inherent uncertainty constraints.

**EAGLE** addresses this by incorporating timestep-advanced token sequences to resolve uncertainty, achieving precise feature-level prediction with minimal overhead.

**Performance**: LLaMA2-Chat 70B achieves 2.7x-3.5x speedup, doubling throughput.

### 3.3 Multi-head Decoding (Medusa)

**Paper**: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"

```
                    [Target Model]
                          |
                    [Base LM Head]
                          |
              +-----+-----+-----+-----+
              |     |     |     |     |
           Medusa_0 Medusa_1 ... Medusa_k
           (predict t+1) (predict t+2)  (predict t+k)
```

**Features**: Adds multiple decoding heads on top of the original model; no separate draft model needed; low training cost (only new heads are trained).

### 3.4 Self-Speculative Decoding

Uses the target model itself as the draft model through:
1. **Early Exit**: Exit at intermediate layers for draft
2. **Layer Skipping**: Skip layers for fast draft
3. **Quantized Draft**: Use low-precision version for draft

**Advantages**: No additional model required; higher memory efficiency; suitable for edge deployment.

**Representative works**: Kangaroo (2024), CATS (2026), SpecMoE (2026), KnapSpec (2026), Quasar (2026).

---

## 4. Verification Mechanisms

### Standard Acceptance Criteria

```
For each position i:
  Acceptance probability = min(1, p_target(xᵢ) / p_draft(xᵢ))
  
Mathematical guarantee: Final distribution = Target model distribution (lossless)
```

### Block Verification

**SpecTr-GBV** (2026): Verifies multiple candidates as a whole rather than per-token, improving efficiency.

### Adaptive Verification

Adaptive strategies for MoE models, optimizing expert invocation during verification.

---

## 5. Training-free vs Training-based Methods

| Category | Method | Advantage | Disadvantage |
|----------|--------|-----------|--------------|
| **Training-free** | External Draft Model | No training, directly usable | Extra model memory |
| **Training-free** | Self-speculative | No additional model | Lower draft quality |
| **Training-based** | Medusa Heads | High-quality draft | Need to train new components |
| **Training-based** | EAGLE | Precise feature-level prediction | Need to train draft network |

---

## 6. Performance Comparison

### 6.1 Speedup Results

| Model | Method | Speedup | Notes |
|-------|--------|---------|-------|
| T5-XXL | Google SD | 2-3x | ICML 2023 original paper |
| Chinchilla 70B | DeepMind SS | 2-2.5x | Distributed setup |
| LLaMA2-Chat 70B | EAGLE | 2.7-3.5x | Feature-level prediction |
| Vicuna series | Medusa | 2-2.5x | Multi-head |
| LLaMA series | SpecInfer | 1.5-2.8x | Tree-based |

### 6.2 Acceptance Rate by Task

| Task Type | Acceptance Rate | Reason |
|-----------|----------------|--------|
| Code generation | 85-95% | Structured output, draft easily matches |
| Math reasoning | 70-80% | Lower, complex reasoning paths |
| General conversation | 80-90% | Moderate complexity |
| Compliance text | 91.3% | Low-entropy text |

**Insight**: Low-entropy, structured tasks have higher acceptance rates and better speedup.

---

## 7. Recent Advances (2025-2026)

### Self-Speculative Decoding
| Paper | Key Technique |
|-------|---------------|
| CATS (2026) | Cascaded Adaptive Tree Speculation |
| SlimSpec (2026) | Low-Rank Draft LM-Head |
| SpecMoE (2026) | Self-Assisted Speculative for MoE |
| KnapSpec (2026) | Adaptive Layer Selection as Knapsack |
| Quasar (2026) | Quantized Self-Speculative |

### Multi-Draft Methods
| Paper | Key Technique |
|-------|---------------|
| UniVer (2026) | Unified Multi-step/Multi-draft |
| SpecTr-GBV (2026) | Multi-Draft Block Verification |
| Hydra (2024) | Sequentially-Dependent Draft Heads |

### Domain-Specific Applications
| Paper | Application |
|-------|------------|
| CoVSpec (2026) | Device-Edge VLM Co-Inference |
| SpecFed (2026) | Federated LLM Inference |
| DiP-SD (2026) | Distributed Pipelined SD at Edge |

---

## 8. Open Source Projects

| Project | Organization | SD Support | Highlights |
|---------|-------------|------------|-----------|
| **vLLM** | UC Berkeley | ✅ Built-in | High-performance inference, PagedAttention |
| **TensorRT-LLM** | NVIDIA | ✅ Built-in | GPU-optimized, production deployment |
| **llama.cpp** | Georgi Gerganov | ✅ Supported | CPU/edge-friendly |
| **MLX-LM** | Apple | ✅ Supported | Apple Silicon optimization |
| **Medusa** | FasterDecoding | ✅ Purpose-built | Multi-head decoding |
| **EAGLE** | SafeAILab | ✅ Purpose-built | Feature-level speculation |
| **SpecInfer/FlexFlow** | CMU | ✅ Purpose-built | Tree-based inference |

---

## 9. Future Directions and Challenges

### Open Challenges

**Technical**:
- **Draft-Target matching**: How to optimally select draft models
- **Acceptance rate optimization**: Low acceptance on complex tasks
- **Long context**: KV cache compatibility with speculative decoding
- **Dynamic load**: Heterogeneous request handling

**Systems**:
- **Memory overhead**: Multi-model/multi-head memory pressure
- **Scheduling complexity**: Multi-tenant optimization
- **Edge deployment**: Feasibility of on-device speculative decoding

### Predicted Directions

1. **Training-aware speculative decoding**: Deep integration with model training
2. **Hardware co-design**: Dedicated accelerator support for SD
3. **Intelligent draft selection**: Context-aware dynamic draft strategies
4. **Multi-modal unification**: Unified speculative frameworks for text/image/video

---

## References

1. Leviathan, Y., et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
2. Chen, C., et al. "Accelerating Large Language Model Decoding with Speculative Sampling." [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
3. Miao, X., et al. "Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification." ASPLOS 2024. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
4. Li, Y., et al. "Speculative Sampling Requires Rethinking Feature Uncertainty." [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
5. Cai, T., et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads."

```bibtex
@article{leviathan2023fast,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  journal={ICML},
  year={2023}
}

@article{chen2023accelerating,
  title={Accelerating Large Language Model Decoding with Speculative Sampling},
  author={Chen, Charlie and others},
  journal={arXiv preprint arXiv:2302.01318},
  year={2023}
}

@inproceedings{miao2024specinfer,
  title={Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification},
  author={Miao, Xupeng and others},
  booktitle={ASPLOS},
  year={2024}
}
```