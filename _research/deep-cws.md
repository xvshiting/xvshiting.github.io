---
name: DEEP-CWS
description: Distilling Efficient pre-trained models with Early exit and Pruning for scalable Chinese Word Segmentation. Achieves over 100x speedup in inference latency while maintaining F1 score of 97.81 on the PKU benchmark.
tags: [NLP, Chinese Word Segmentation, Model Compression]
url_paper: https://doi.org/10.1016/j.ins.2025.122470
github:
status: published
type: journal
venue: "Information Sciences"
year: 2025
highlight: true
---

## Abstract

Chinese Word Segmentation (CWS) is essential for a broad spectrum of NLP tasks. However, the high inference cost of large pre-trained models restricts their scalability. DEEP-CWS distills pre-trained transformer models into lightweight CNNs, incorporating pruning, early exit mechanisms, and ONNX optimization to improve inference speed significantly — achieving over 100x speedup without compromising segmentation quality.

## Citation

```bibtex
@article{xu2025deepcws,
  title={DEEP-CWS: Distilling Efficient pre-trained models with Early exit and Pruning for scalable Chinese Word Segmentation},
  author={Xu, Shiting},
  journal={Information Sciences},
  volume={719},
  pages={122470},
  year={2025},
  publisher={Elsevier}
}
```