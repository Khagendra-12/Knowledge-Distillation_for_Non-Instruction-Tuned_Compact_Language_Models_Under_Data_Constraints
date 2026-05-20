<div align="center">

# Knowledge Distillation for Non-Instruction-Tuned Compact Language Models Under Data Constraints

### Impact of Pre-Distillation Fine-Tuning on Student Model Performance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue?style=flat-square)](https://ieee.org)

**Teacher:** `mistralai/Mistral-7B-Instruct-v0.1` (7.24B params) &nbsp;→&nbsp; **Student:** `Locutusque/TinyMistral-248M` (248M params)

**15.13× Compression** &nbsp;|&nbsp; **~3× Faster Inference** &nbsp;|&nbsp; **83–84% BERTScore Retention**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [Models](#models)
  - [Dataset](#dataset)
  - [Pre-Distillation Fine-Tuning (SFT Stage)](#pre-distillation-fine-tuning-sft-stage)
  - [Distillation Pipeline](#distillation-pipeline)
  - [Loss Function](#loss-function)
- [Experimental Conditions](#experimental-conditions)
- [Results](#results)
  - [Table 1 — Full Results Across All Conditions](#table-1--full-results-across-all-conditions)
  - [Table 2 — Percentage Change vs. Baseline](#table-2--percentage-change-vs-baseline)
  - [Table 3 — KD-Specific Results](#table-3--kd-specific-results)
  - [Key Findings](#key-findings)
- [Hyperparameters](#hyperparameters)
- [Installation & Usage](#installation--usage)
  - [Installation](#installation)
  - [Running the Baseline](#running-the-baseline)
  - [Running Pre-Distillation Fine-Tuning](#running-pre-distillation-fine-tuning)
  - [Running Distillation](#running-distillation)
  - [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Overview

This repository contains the full implementation for the research paper:

> **"Knowledge Distillation for Non-Instruction-Tuned Compact Language Models Under Data Constraints & Impact of Pre-Distillation Fine-Tuning"**

Large language models (LLMs) achieve strong performance across NLP tasks but are prohibitively expensive to deploy on resource-constrained hardware. **Knowledge Distillation (KD)** offers a principled way to transfer capabilities from a large *teacher* model to a compact *student* model. However, most prior work assumes access to large instruction-following datasets, and the effect of pre-distillation fine-tuning on non-instruction-tuned base models remains underexplored.

This project systematically investigates:

1. Whether KD can be successfully applied to a **non-instruction-tuned compact base model** under constrained data conditions
2. Whether a **pre-distillation fine-tuning (SFT) stage** improves the quality of the distilled student
3. Whether the **softness of that pre-FT stage** (standard hard-label SFT vs. label-smoothed SFT) makes a measurable difference

All experiments use a fixed 10,000-sample budget from the [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset to simulate real-world data constraints.

---

## Research Questions

| # | Question |
|---|----------|
| RQ1 | Does knowledge distillation improve a non-instruction-tuned compact LM when applied directly without any prior fine-tuning? |
| RQ2 | Does a pre-distillation supervised fine-tuning (SFT) stage improve distillation quality compared to KD without pre-FT? |
| RQ3 | Does the softness of pre-distillation FT (label smoothing + smaller LoRA rank) further improve KD outcomes vs. standard hard-label SFT? |
| RQ4 | What is the practical efficiency trade-off between distillation quality and inference speed/model size? |

---

## Repository Structure

```
KnowledgeDistillation-under-data-constraints/
│
├── Baseline/
│   ├── config.py               # Baseline & SFT hyperparameters
│   ├── dataset.py              # Dolly-15k loader with 90/10 split
│   ├── utils.py                # Tokenizer utilities
│   ├── finetune.py             # SFT training (LoRA r=8, hard labels)
│   ├── finetune_soft.py        # SFT-Soft training (LoRA r=4, label smoothing=0.2)
│   ├── merge_lora.py           # Merges LoRA adaptors → base model (SFT)
│   ├── merge_lora_soft.py      # Merges LoRA adaptors → base model (SFT-Soft)
│   ├── baseline_eval.py        # Evaluates raw TinyMistral-248M
│   ├── eval_sft.py             # Evaluates SFT-merged checkpoint
│   └── eval_sft_soft.py        # Evaluates SFT-Soft-merged checkpoint
│
├── Distillation_Pipeline/
│   ├── config.py               # KD hyperparameters (τ, α, epochs, etc.)
│   ├── train_kd.py             # KD training: SFT-initialised student
│   ├── train_kd_base.py        # KD training: raw base model student (no pre-FT)
│   ├── train_kd_soft.py        # KD training: SFT-Soft-initialised student
│   ├── eval_kd.py              # Evaluates KD + SFT distilled model
│   ├── eval_kd_base.py         # Evaluates BaseModel KD distilled model
│   └── eval_kd_soft.py         # Evaluates KD + SFT-Soft distilled model
│
├── Load Models/
│   ├── LoadTeacher.py          # Loads Mistral-7B in 4-bit NF4 quantisation
│   └── LoadStudent.py          # Loads TinyMistral-248M
│
├── Results.txt                 # Full evaluation results for all 6 conditions
├── requirments.txt             # Python dependencies
└── README.md
```

---

## Methodology

### Models

| Role | Model | Parameters | Notes |
|------|-------|-----------|-------|
| **Teacher** | `mistralai/Mistral-7B-Instruct-v0.1` | 7.24B | Instruction-tuned; loaded in **4-bit NF4** quantisation via BitsAndBytes; **fully frozen** during distillation |
| **Student** | `Locutusque/TinyMistral-248M` | 248M | Decoder-only causal transformer; same Mistral tokeniser family; trained in **float32** |

**Compression ratio: 15.13×**

### Dataset

| Property | Value |
|----------|-------|
| Name | `databricks/databricks-dolly-15k` |
| Type | Instruction-following (instruction, optional context, response) |
| Train / Val split | 90% / 10% — `seed=42` |
| Training samples | 10,000 (first 10k after split — data constraint simulation) |
| PPL evaluation | Full validation set (~1,500 examples) |
| Generation evaluation | 100-sample subset of validation |
| Max sequence length | 512 tokens |
| Prompt format | Alpaca-style (`### Instruction / ### Input / ### Response`) |

### Pre-Distillation Fine-Tuning (SFT Stage)

A key contribution of this work is the **pre-distillation fine-tuning stage** — supervised fine-tuning of the student on a domain-matched corpus *before* distillation begins. This reduces the distribution gap between the student and teacher at the start of distillation.

Two SFT variants are explored:

#### Standard SFT (`finetune.py`)
- **LoRA** (Low-Rank Adaptation): rank `r=8`, alpha=16
- Target modules: `q_proj`, `v_proj`
- Hard-label cross-entropy loss (no label smoothing)
- 1 epoch, `lr=1e-4`, cosine scheduler, warmup=100 steps

#### Soft SFT (`finetune_soft.py`)
- **LoRA**: rank `r=4`, alpha=8 — lighter adaptation
- Target module: `q_proj` only
- **Label smoothing factor: 0.2** — softens the hard-label signal
- 1 epoch, `lr=1e-4`, cosine scheduler, warmup=30 steps

After training, LoRA adaptors are **merged** into the base model weights via `merge_lora.py` / `merge_lora_soft.py`, producing a standard dense checkpoint for distillation.

### Distillation Pipeline

The distillation uses a custom `LogitKDTrainer` (extending HuggingFace `Trainer`) that overrides `compute_loss()` to implement token-level forward KL distillation.

**Pipeline (per training step):**

```
1. Forward pass: student logits  z_S  [float32]
2. Forward pass: teacher logits  z_T  [no_grad, bf16, 4-bit NF4]
3. Causal shift:  z_S = z_S[:,  :-1, :]   |   labels = labels[:, 1:]
4. Mask:          select only response token positions (labels ≠ -100)
5. Vocab truncate: V_eff = min(|V_S|, |V_T|)
6. Soft dists:    log_p_S = log_softmax(z_S / τ)   |   p_T = softmax(z_T / τ)   [τ=4.5]
7. Token KL:      kl = mean( sum( p_T × (log p_T - log p_S), dim=-1 ) )
8. Scale KL:      L_KL = kl × τ²  =  kl × 20.25
9. CE loss:       L_CE = student cross-entropy on response tokens
10. Combine:      L = 0.1 × L_CE  +  0.9 × L_KL
```

### Loss Function

The combined distillation loss is:

$$\mathcal{L}_{\text{total}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{KL}}$$

where:

$$\mathcal{L}_{\text{KL}} = \tau^2 \cdot \sum_{v \in V_{\text{eff}}} p_T^\tau(v) \left[ \log p_T^\tau(v) - \log p_S^\tau(v) \right]$$

$$p_T^\tau(v) = \text{softmax}(z_T / \tau), \quad p_S^\tau(v) = \text{softmax}(z_S / \tau)$$

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature `τ` | 4.5 | Higher temperature softens distributions, providing richer gradient signal to the student |
| Alpha `α` | 0.9 | 90% of gradient from teacher distributions — student primarily learns to mimic teacher |
| τ² scaling | 20.25 | Compensates for gradient magnitude reduction from τ division (Hinton et al., 2015) |
| KL type | Forward KL — KL(p_T ‖ p_S) | Mode-covering; student penalised for missing mass where teacher is confident |

> **Note on token masking:** The KL loss is computed **only on response token positions**. Prompt tokens are masked with `-100` and excluded from both the CE and KL computations. This ensures the student learns to generate responses, not reproduce prompts.

---

## Experimental Conditions

Six conditions are evaluated in total — three baselines (no distillation) and three distillation variants:

| Condition | Script | Student Initialisation | Description |
|-----------|--------|----------------------|-------------|
| **Baseline** | `baseline_eval.py` | `TinyMistral-248M` (raw) | Raw model with zero training — quality floor |
| **SFT** | `finetune.py` → `eval_sft.py` | LoRA r=8, hard labels, 1 epoch | Hard-label supervised fine-tuning only, no KD |
| **SFT-Soft** | `finetune_soft.py` → `eval_sft_soft.py` | LoRA r=4, label smooth=0.2, 1 epoch | Label-smoothed supervised fine-tuning only, no KD |
| **BaseModel KD** | `train_kd_base.py` → `eval_kd_base.py` | `TinyMistral-248M` (raw) | KD applied directly to base model — **ablates pre-FT** |
| **KD + SFT** | `train_kd.py` → `eval_kd.py` | SFT-merged checkpoint | KD after standard hard-label pre-FT |
| **KD + SFT-Soft** | `train_kd_soft.py` → `eval_kd_soft.py` | SFT-Soft-merged checkpoint | KD after soft pre-FT — **best PPL among KD variants** |

---

## Results

Evaluation uses the full validation set for **perplexity** and a 100-sample subset for **generation metrics** (ROUGE, BLEU, METEOR, BERTScore). BERTScore is computed with `roberta-large` on CPU.

---

### Table 1 — Full Results Across All Conditions

> ↓ = lower is better &nbsp;|&nbsp; ↑ = higher is better

| Condition | PPL ↓ | KL Div ↓ | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU ↑ | METEOR ↑ | BERTScore ↑ | Latency ↓ | Speed ↑ |
|-----------|------:|--------:|----------:|----------:|----------:|-------:|---------:|------------:|----------:|--------:|
| **Baseline** | 35.17 | — | 0.077 | 0.016 | 0.070 | 0.010 | 0.075 | 0.783 | 2.33 s | 55.09 t/s |
| **SFT** | **31.56** | — | 0.147 | 0.053 | 0.133 | 0.008 | 0.109 | 0.808 | 2.76 s | 46.37 t/s |
| **SFT-Soft** | 36.40 | — | 0.086 | 0.022 | 0.079 | 0.016 | 0.074 | 0.768 | 2.55 s | 50.33 t/s |
| **BaseModel KD** | 39.06 | 2.171 | 0.180 | 0.057 | 0.146 | 0.013 | 0.127 | 0.837 | — | — |
| **KD + SFT** | 40.37 | 2.209 | **0.189** | **0.065** | **0.158** | 0.013 | **0.143** | 0.830 | **0.90 s** | 50.68 t/s |
| **KD + SFT-Soft** | 38.68 | **2.175** | 0.177 | 0.059 | 0.147 | 0.013 | 0.133 | **0.837** | 0.94 s | 50.03 t/s |

*Bold = best value in each column.*

---

### Table 2 — Percentage Change vs. Baseline

All values show **percentage change relative to the Baseline** condition. `—` indicates the baseline reference value.

> For **PPL and Latency** (lower is better): a **negative %** means improvement. For all other metrics (higher is better): a **positive %** means improvement.

| Condition | PPL ↓ | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU ↑ | METEOR ↑ | BERTScore ↑ | Latency ↓ | Speed ↑ |
|-----------|------:|----------:|----------:|----------:|-------:|---------:|------------:|----------:|--------:|
| **Baseline** | — | — | — | — | — | — | — | — | — |
| **SFT** | **−10.26%** | +91.91% | +240.65% | +91.52% | −12.50% | +45.27% | +3.18% | +18.45% ⚠️ | −15.82% |
| **SFT-Soft** | +3.51% | +12.14% | +38.71% | +12.93% | +67.71% | −1.07% | −1.92% | +9.44% ⚠️ | −8.65% |
| **BaseModel KD** | +11.09% | +134.86% | +270.32% | +109.91% | +35.42% | +69.51% | +6.90% | — | — |
| **KD + SFT** | +14.81% | **+146.21%** | **+318.71%** | **+126.72%** | +32.29% | **+90.01%** | +6.04% | **−61.37%** | −8.01% |
| **KD + SFT-Soft** | +10.00% | +131.20% | +281.29% | +111.06% | +32.29% | +76.83% | **+6.94%** | −59.66% | −9.19% |

> ⚠️ SFT and SFT-Soft show *increased* latency vs. baseline — the LoRA fine-tuning process slightly increases generation cost at inference due to longer, more coherent outputs. KD models are significantly faster because of 15× model compression.

**Reading the table:**
- **KD + SFT** delivers the largest gains on generation quality — ROUGE-1 up **+146%**, ROUGE-2 up **+319%**, METEOR up **+90%** vs. baseline
- **KD + SFT-Soft** achieves the best BERTScore improvement (+6.94%) and the lowest KL divergence (2.175), suggesting the best teacher-alignment among KD variants
- The **latency gain from KD** is dramatic: KD + SFT runs at 0.90 s/prompt vs. 2.33 s/prompt baseline — a **61.4% latency reduction**
- SFT alone improves PPL (−10.26%) but all KD variants trade raw PPL for substantially superior generation quality

---

### Table 3 — KD-Specific Results

Comparison across the three distillation conditions only, showing the impact of the pre-distillation fine-tuning strategy.

| KD Condition | Pre-FT Stage | PPL ↓ | KL Div ↓ | ROUGE-1 ↑ | ROUGE-L ↑ | METEOR ↑ | BERTScore ↑ | Latency ↓ | Compression |
|--------------|-------------|------:|---------:|----------:|----------:|---------:|------------:|----------:|:-----------:|
| **BaseModel KD** | None (raw base) | 39.06 | 2.171 | 0.180 | 0.146 | 0.127 | 0.837 | — | 15.13× |
| **KD + SFT** | Hard-label LoRA r=8 | 40.37 | 2.209 | **0.189** | **0.158** | **0.143** | 0.830 | **0.90 s** | 15.13× |
| **KD + SFT-Soft** | Soft LoRA r=4 + smooth=0.2 | **38.68** | **2.175** | 0.177 | 0.147 | 0.133 | **0.837** | 0.94 s | 15.13× |

**Δ: SFT-Soft KD vs. BaseModel KD (pre-FT impact):**

| Metric | BaseModel KD | KD + SFT-Soft | Δ |
|--------|------------:|------------:|--:|
| PPL ↓ | 39.06 | **38.68** | **−0.38 (−0.97%)** |
| KL Div ↓ | 2.171 | **2.175** | +0.004 |
| ROUGE-1 ↑ | 0.180 | 0.177 | −0.003 |
| ROUGE-L ↑ | 0.146 | 0.147 | +0.001 |
| METEOR ↑ | 0.127 | 0.133 | **+0.006 (+4.72%)** |
| BERTScore ↑ | 0.837 | **0.837** | = (tied best) |

---

### Key Findings

**1. KD models dramatically outperform their pre-FT-only counterparts on generation quality.** Despite higher perplexity, all distilled variants more than double ROUGE-1 and METEOR scores compared to the raw baseline. BERTScore improves from 0.783 → 0.837, indicating stronger semantic alignment with reference answers.

**2. Pre-distillation fine-tuning helps — and softer pre-FT helps slightly more.** KD + SFT-Soft achieves the best perplexity among all KD conditions (38.68 vs. 39.06 for BaseModel KD), suggesting that a lighter, smoother initialisation before distillation produces a marginally better-calibrated student.

**3. The PPL–generation quality divergence is a key finding.** SFT achieves the lowest perplexity (31.56) but the worst generation metrics among trained conditions. All KD variants have higher PPL but substantially better ROUGE/METEOR/BERTScore. This demonstrates that **perplexity alone is an insufficient proxy** for instruction-following quality.

**4. Distillation provides ~3× faster inference at 15× compression.** KD + SFT achieves 0.90 s/prompt vs. 2.76 s/prompt for SFT (3.07× speedup) with a BERTScore of 0.830 vs. 0.808 — better semantic quality at dramatically lower latency.

**5. Forward KL with high α=0.9 and τ=4.5 is effective in the low-data regime.** The student primarily learns from teacher distribution soft targets rather than hard labels, which appears beneficial when training data is constrained to 10,000 samples.

---

## Hyperparameters

### SFT Stage

| Parameter | SFT (`finetune.py`) | SFT-Soft (`finetune_soft.py`) |
|-----------|--------------------:|------------------------------:|
| Learning rate | 1e-4 | 1e-4 |
| Epochs | 1 | 1 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 (effective batch = 16) | 4 (effective batch = 16) |
| Warmup steps | 100 | 30 |
| Weight decay | 0.01 | 0.01 |
| LR scheduler | cosine | cosine |
| Max grad norm | 1.0 | 1.0 |
| Label smoothing | 0.0 | **0.2** |
| LoRA rank (r) | **8** | **4** |
| LoRA alpha | 16 | 8 |
| LoRA target modules | q_proj, v_proj | **q_proj only** |
| LoRA dropout | 0.05 | 0.05 |
| Precision | bf16 (auto) | bf16 (auto) |

### Distillation Stage (all three KD scripts)

| Parameter | Value |
|-----------|-------|
| Temperature (τ) | **4.5** |
| Alpha (α) | **0.9** |
| τ² scaling factor | 20.25 |
| KL type | Forward KL |
| Learning rate | 1e-4 |
| Epochs | 7 |
| Batch size (per device) | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| Warmup steps | 100 |
| Weight decay | 0.05 |
| LR scheduler | cosine |
| Max grad norm | 1.0 |
| Student precision | float32 |
| Teacher loading | 4-bit NF4 (BitsAndBytes) |
| Teacher state | Frozen (eval mode, no grad) |
| Max sequence length | 512 tokens |

---

## Installation & Usage

### Installation

```bash
git clone https://github.com/Khagendra-12/KnowledgeDistillation-under-data-constraints.git
cd KnowledgeDistillation-under-data-constraints
pip install -r requirments.txt
```

> **Hardware requirements:** A CUDA-capable GPU with at least **16GB VRAM** is required to load the teacher model in 4-bit NF4 quantisation alongside the student during distillation. The baseline and SFT stages can run on smaller GPUs (~8GB).

> **HuggingFace access:** You will need to be logged in to HuggingFace (`huggingface-cli login`) to download `mistralai/Mistral-7B-Instruct-v0.1`, which requires accepting the model license on the HuggingFace model page.

---

### Running the Baseline

Evaluate the raw `TinyMistral-248M` model with no training:

```bash
cd Baseline
python baseline_eval.py
```

---

### Running Pre-Distillation Fine-Tuning

**Standard SFT** (LoRA r=8, hard labels):
```bash
cd Baseline
python finetune.py
python merge_lora.py        # merges LoRA adaptors into base weights
python eval_sft.py          # evaluate the SFT checkpoint
```

**Soft SFT** (LoRA r=4, label smoothing=0.2):
```bash
cd Baseline
python finetune_soft.py
python merge_lora_soft.py   # merges soft LoRA adaptors into base weights
python eval_sft_soft.py     # evaluate the SFT-Soft checkpoint
```

> Merged checkpoints are saved to `Baseline/outputs/tinymistral-sft-merged` and `Baseline/outputs/tinymistral-sft-merged-soft` respectively. These are the starting points for distillation.

---

### Running Distillation

**KD on raw base model** (no pre-distillation FT — ablation condition):
```bash
cd Distillation_Pipeline
python train_kd_base.py
```

**KD with SFT-initialised student:**
```bash
cd Distillation_Pipeline
python train_kd.py
```

**KD with SFT-Soft-initialised student:**
```bash
cd Distillation_Pipeline
python train_kd_soft.py
```

> All three scripts share the same `LogitKDTrainer` and `compute_loss()` implementation. They differ only in which student checkpoint they load, controlled via `config.py` (`STUDENT_MODEL_B`, `STUDENT_MODEL`, `STUDENT_MODEL_S`).

---

### Evaluation

```bash
cd Distillation_Pipeline

python eval_kd_base.py      # evaluate BaseModel KD
python eval_kd.py           # evaluate KD + SFT
python eval_kd_soft.py      # evaluate KD + SFT-Soft
```

Each evaluation script computes:
- **Perplexity** — over the full validation set
- **KL Divergence** — between student and teacher on validation set
- **ROUGE-1/2/L** — n-gram overlap with reference responses
- **BLEU** — precision-based n-gram metric
- **METEOR** — recall-aware generation metric
- **BERTScore F1** — semantic similarity via `roberta-large` (runs on CPU)
- **Inference latency** and **throughput** (tokens/s)
- **Compression ratio** (teacher / student parameter count)

---

## Limitations

- **Single architecture family:** All experiments use the Mistral model family for both teacher and student. Generalisation to other architectures (e.g., LLaMA, GPT-2 family) is not evaluated.
- **Single dataset:** Experiments use Databricks Dolly-15k only (general instruction following). Domain-specific or code/reasoning datasets may produce different results.
- **Perplexity metric mismatch:** Evaluation perplexity is computed over the full sequence (prompt + response), while training loss is computed over response tokens only. This creates a slight measurement inconsistency that may disadvantage KD models in PPL comparison.
- **No downstream benchmark evaluation:** Results are not evaluated on standard benchmarks such as HellaSwag, WinoGrande, or MMLU.
- **Label smoothing scope:** The `label_smoothing_factor=0.2` in SFT-Soft applies only to the pre-FT stage — it does not affect the distillation loss in `train_kd_soft.py`.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{khagendra2025kd,
  title     = {Knowledge Distillation for Non-Instruction-Tuned Compact Language Models
               Under Data Constraints and Impact of Pre-Distillation Fine-Tuning},
  author    = {Khagendra},
  journal   = {IEEE},
  year      = {2025},
  note      = {Manuscript in preparation}
}
```

### References

This work builds on the following prior research:

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108
- Jiao, X. et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding.* arXiv:1909.10351
- Gu, Y. et al. (2024). *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR 2024. arXiv:2306.08543
- Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Li, J. et al. (2024). *Learning with Less: Knowledge Distillation from LLMs via Unlabeled Data.* arXiv:2411.08028
- Gu, H. et al. (2024). *Pre-training Distillation for LLMs: A Design Space Exploration.* arXiv:2410.16215

---

<div align="center">



</div>
