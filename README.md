# Listen First, Then Answer: Timestamp-Grounded Speech Reasoning

**Anonymous submission to Interspeech 2026**

<p align="center">
  <img src="static/images/fig1_teaser.pdf" width="90%">
</p>

> **TL;DR**: We propose a timestamp-grounded reasoning framework for Large Audio-Language Models (LALMs) that anchors each reasoning step to specific temporal segments of the input audio, improving both task accuracy and reasoning interpretability.

---

## Abstract

Large audio-language models (LALMs) can generate reasoning chains for their predictions, but it remains unclear whether these reasoning chains remain grounded in the input audio. In this paper, we propose an RL-based strategy that grounds the reasoning outputs of LALMs with explicit timestamp annotations referring to relevant segments of the audio signal. Our analysis shows that timestamp grounding leads the model to attend more strongly to audio tokens during reasoning generation. Experiments on four speech-based benchmark datasets demonstrate that our approach improves performance compared to both zero-shot reasoning and fine-tuning without timestamp grounding. Additionally, grounding amplifies desirable reasoning behaviors, such as region exploration, audiology verification, and consistency, underscoring the importance of grounding mechanisms for faithful multimodal reasoning.

**Index Terms**: Large Audio Language Models, Reasoning, Grounding, Interpretability

---

## Method

<p align="center">
  <img src="static/images/fig2_main_figure.pdf" width="95%">
</p>

Our framework consists of a **two-stage training pipeline**:

### Stage 1: Supervised Timestamp Alignment (STA)

We first build a strong temporal grounding primitive by training the model on timestamp prediction tasks.

- Construct a timestamp-annotated speech corpus (~268k examples) using [`whisper-timestamped`](https://github.com/linto-ai/whisper-timestamped), which produces word-level timestamps via Dynamic Time Warping
- Convert timestamped transcripts into Q&A-style grounding data, where the model learns to predict start/end timestamps for given spoken sentences
- Datasets: LibriSpeech, CoVoST 2, MELD, multi-speaker dialogue, YouTube8M speech data

### Stage 2: Timestamp-Grounded Reasoning (GRPO)

We adopt Group Relative Policy Optimization (GRPO) with two complementary rewards:

- **Answer Correctness Reward** (*R_answer*): Binary reward (1.0 if prediction matches ground truth, 0.0 otherwise)
- **Timestamp Grounding Reward** (*R_tg*): Compaction score that encourages concise yet grounded reasoning — penalizes both missing grounding signals and overly verbose timestamp usage
- **Final Reward**: *R = R_answer + R_tg*
- Training data: ~47k examples from MELD, multi-speaker dialogue QA, and YouTube8M speech QA

---

## Key Findings

### 1. Current LALMs Don't Truly Listen

We conduct a semantic-based attention analysis and find that baseline LALMs allocate only a small fraction of attention to audio tokens. System tokens dominate attention allocation, receiving **15x more attention** than audio tokens on a per-token basis — even when producing correct answers.

### 2. Grounding Is Listening

Our timestamp-aligned model allocates **significantly higher attention to audio tokens** across transformer layers compared to the vanilla model, with the most pronounced increase observed during the reasoning stage.

### 3. Grounding Improves Reasoning Quality

| Model | Regions Explored | Audiology Verify | Consistency | Acc. |
|:------|:----------------:|:----------------:|:-----------:|:----:|
| Zero-Shot (Standard CoT) | 1.3 | 0.27 | 0.72 | 69.4 |
| Ours | 1.8 | 0.56 | 0.83 | 74.5 |

Timestamp grounding reshapes reasoning behavior: the model explores more distinct audio regions, achieves higher audiology verification scores, and produces more consistent reasoning chains.

---

## Results

### Benchmark Comparison

Performance comparison on MMAU-mini-Speech, MMAR-Speech, AIR-Bench, and MELD:

| Methods | Size | MMAU-mini Speech (%) | MMAR-Speech (%) | AIR-Bench SER (%) | AIR-Bench SNV (%) | AIR-Bench SIC (%) | MELD (%) |
|:--------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| *Proprietary Models* | | | | | | | |
| Gemini-2.5-Flash | – | **75.08** | 72.41 | 56.4 | 68.5 | 83.6 | 61.5 |
| GPT-4o Audio | – | 66.67 | 20.41 | 51.2 | 61.6 | **89.3** | 62.5 |
| *Open-source Models* | | | | | | | |
| SALMONN | 7B | 26.43 | 24.35 | 29.0 | 34.3 | 42.3 | 37.2 |
| Audio Flamingo 3 | 7B | 66.37 | 57.48 | 59.5 | **76.8** | 79.6 | 58.5 |
| *Audio Reasoning Methods* | | | | | | | |
| Audio-CoT | 8.4B | 55.26 | 54.01 | – | – | – | – |
| Audio-Reasoner | 8.4B | 66.07 | 32.99 | 60.5 | 56.3 | 88.1 | 63.2 |
| Audio-Thinker | 8.4B | 73.37 | 64.29 | 56.2 | 67.5 | – | – |
| *Our Ablation Variants* | | | | | | | |
| Qwen2.5-Omni (baseline) | 7B | 70.60 | 59.86 | 60.2 | 63.9 | 83.5 | 60.3 |
| + Only STA | 7B | 71.37 | 61.22 | 59.5 | 66.0 | 84.3 | 62.8 |
| + Reasoning SFT | 7B | 74.47 | 62.93 | 58.5 | 68.1 | 85.0 | 61.8 |
| **Ours (Full)** | **7B** | 74.47 | **64.63** | **62.5** | 70.4 | 89.3 | **64.6** |

Our full model achieves the best overall performance across all benchmarks, even outperforming proprietary models on AIR-Bench and MELD.

---

## Implementation Details

| Config | Value |
|:-------|:------|
| Base Model | Qwen2.5-Omni (7B) |
| GPUs | 8 x H200 |
| Batch Size | 8 per GPU, 4 gradient accumulation steps |
| Learning Rate (Stage 1) | 2 x 10⁻⁵ |
| Learning Rate (Stage 2) | 5 x 10⁻⁶ |
| Sampling Temperature | 0.8 |
| GRPO Responses | 8 per step |
| KL Coefficient (β) | 0.04 |
| Stage 1 Data | ~268k examples |
| Stage 2 Data | ~47k examples |

---

## Citation

```bibtex
@inproceedings{anonymous2026listen,
  title={Listen First, Then Answer: Timestamp-Grounded Speech Reasoning},
  author={Anonymous},
  booktitle={Proc. Interspeech 2026},
  year={2026}
}
```

---

## Acknowledgements

This page was built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
