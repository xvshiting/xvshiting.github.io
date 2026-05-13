---
name: "VIMAR: Vision-Language Informed Malware Analysis and Reasoning Model"
description: We propose VIMAR, a unified vision-language model that supports malware family classification, similarity detection, and open-world analysis via explanation-rich supervision and a two-stage training pipeline. On the Malimg dataset, VIMAR achieves 94.2% accuracy in family classification, surpassing the best CNN baseline by +3.1%. It also attains 85.2% and 88.0% accuracy in zero-shot and few-shot settings.
tags: [Deep Learning, Malware Analysis, Vision-Language Model]
url_paper: https://link.springer.com/article/10.1186/s42400-025-00481-3
github:
status: published
type: journal
venue: "Cybersecurity"
year: 2026
highlight: true
---

## Key Contributions

- We formulate a unified vision-language framework that supports five key malware analysis tasks using a single model architecture.
- We design and generate a high-quality, multi-task, explanation-enhanced dataset aligned with malware-specific visual patterns and reasoning needs.
- We demonstrate that VIMAR achieves competitive performance across tasks, with strong generalization and interpretability, matching or surpassing task-specific baselines.

## Methodology

VIMAR is built upon the SmolVLM architecture, a compact 2.2B parameter vision-language backbone. It takes grayscale byteplot images of malware binaries as input and adapts to a range of task settings via prompt-based conditioning:

- **Classification (CLS)**: Assign a malware image to a predefined family
- **Similarity Classification (SC)**: Determine if two malware images belong to the same family
- **Similarity Preference (SP)**: Decide which of two references is more similar to a query
- **Zero-shot Classification (ZSC)**: Classify into unseen families using textual descriptions
- **Few-shot Classification (FSC)**: Classify with minimal labeled examples

A two-stage training strategy is employed: Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimization (GRPO) to enhance both task performance and output quality.

## Results

| Task | Metric | Score |
|------|--------|-------|
| Family Classification (CLS) | Accuracy | 94.2% |
| Zero-shot Classification (ZSC) | Accuracy | 85.2% |
| Few-shot Classification (FSC) | Accuracy | 88.0% |

## Citation

```bibtex
@article{xu2026vimar,
  title={VIMAR: vision-language informed malware analysis and reasoning model},
  author={Xu, Shiting},
  journal={Cybersecurity},
  volume={9},
  pages={49},
  year={2026},
  publisher={Springer}
}
```