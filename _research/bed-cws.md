---
name: "BED: Chinese Word Segmentation Model Based on Boundary-Enhanced Decoder"
description: A novel Boundary-Enhanced Decoder (BED) for Chinese Word Segmentation that improves Average-F1 by 0.05% and OOV Average-F1 by 0.69%, particularly effective for out-of-vocabulary words.
tags: [NLP, Chinese Word Segmentation, Deep Learning, Transformer]
url_paper: https://doi.org/10.1145/3654823.3654872
github:
status: published
type: conference
venue: "CACML 2024"
year: 2024
---

## Abstract

Chinese Word Segmentation (CWS) is a critical initial step in the Chinese NLP pipeline. Recent advancements in deep learning and pre-training language models have significantly improved CWS performance. Nevertheless, poor performance on Out-Of-Vocabulary (OOV) words remains a challenge. Existing CWS approaches primarily focus on optimizing the encoder, with little attention given to enhancing the decoder. This paper presents BED, a Boundary-Enhanced Decoder for CWS that brings a 0.05% improvement on Average-F1 and a 0.69% improvement on OOV Average-F1.

## Key Idea

Inspired by the human process of word segmentation — where easy parts are split first, and challenging parts are handled with additional context — BED employs a two-module approach:

1. **Boundary Detection Module**: Identifies whether a character resides at the starting position of a word, producing a binary classification for coarse-grained segmentation
2. **Multi-Grained Decoder Module**: Uses the boundary detection output to create an attention mask that restricts each token to only attend within its segment, enabling fine-grained segmentation

## Results

| Model | PKU F1 | MSR F1 | AS F1 | CITYU F1 | Avg F1 | Avg OOV F1 |
|-------|--------|--------|-------|----------|--------|-------------|
| BERT+softmax | 96.56 | 98.44 | 96.71 | 97.88 | 97.39 | 84.82 |
| BERT+softmax+BED | **96.71** | **98.46** | 96.69 | **97.91** | **97.44** | **85.51** |
| WMSeg+crf+BED | **96.76** | **98.56** | **96.79** | **98.04** | **97.53** | **85.50** |

## Citation

```bibtex
@inproceedings{xu2024bed,
  title={BED: Chinese Word Segmentation Model Based on Boundary-Enhanced Decoder},
  author={Xu, Shiting},
  booktitle={2024 3rd Asia Conference on Algorithms, Computing and Machine Learning (CACML)},
  year={2024},
  pages={1--8},
  publisher={ACM},
  doi={10.1145/3654823.3654872}
}
```