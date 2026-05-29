````md
# Knowledge Distillation for Non-Instruction-Tuned Compact Language Models Under Data Constraints

## Impact of Pre-Distillation Fine-Tuning on Student Model Performance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue?style=flat-square)](https://ieee.org)

> Official implementation accompanying the research paper:
>
> **"Knowledge Distillation for Non-Instruction-Tuned Compact Language Models Under Data Constraints & Impact of Pre-Distillation Fine-Tuning"**

---

## Overview

Large Language Models (LLMs) demonstrate strong instruction-following and generation capabilities but remain impractical for deployment in resource-constrained environments due to their high computational and memory requirements.

This project investigates whether **Knowledge Distillation (KD)** can effectively transfer instruction-following behaviour from a large instruction-tuned teacher model to a compact **non-instruction-tuned** student model under a **strict data budget of only 10,000 samples**.

The work specifically studies:

1. Whether KD can improve a raw non-instruction-tuned compact language model
2. Whether a pre-distillation supervised fine-tuning (SFT) stage improves distillation quality
3. Whether a softer pre-distillation setup (label smoothing + lighter LoRA adaptation) improves downstream KD outcomes
4. The trade-off between semantic generation quality, perplexity, compression ratio, and inference efficiency

The experiments use:

- **Teacher:** `mistralai/Mistral-7B-Instruct-v0.1` (7.24B)
- **Student:** `Locutusque/TinyMistral-248M` (248M)
- **Compression Ratio:** `15.13×`
- **Dataset:** `databricks/databricks-dolly-15k`
- **Distillation Objective:** Forward KL logit distillation

---

## Key Contributions

- Systematic evaluation of KD on a **non-instruction-tuned compact causal LM**
- Controlled comparison of:
  - Direct KD on raw base model
  - KD after standard SFT
  - KD after soft label-smoothed SFT
- Distillation under a **strict 10k-sample data constraint**
- Empirical analysis of the divergence between **Perplexity** and **generation quality metrics**
- Public release of:
  - Training code
  - Evaluation scripts
  - Experimental configuration files
  - Distillation pipeline implementation

---

## Research Questions

| ID | Research Question |
|---|---|
| RQ1 | Can knowledge distillation improve a non-instruction-tuned compact language model under limited data conditions? |
| RQ2 | Does pre-distillation supervised fine-tuning improve distillation quality compared to direct KD on the base model? |
| RQ3 | Does softer pre-distillation fine-tuning (label smoothing + lighter LoRA adaptation) improve downstream KD behaviour? |
| RQ4 | What are the practical efficiency gains in latency and compression achieved through distillation? |

---

## Repository Structure

```text
KnowledgeDistillation-under-data-constraints/
│
├── Baseline/
│   ├── config.py
│   ├── dataset.py
│   ├── utils.py
│   ├── finetune.py
│   ├── finetune_soft.py
│   ├── merge_lora.py
│   ├── merge_lora_soft.py
│   ├── baseline_eval.py
│   ├── eval_sft.py
│   └── eval_sft_soft.py
│
├── Distillation_Pipeline/
│   ├── config.py
│   ├── train_kd.py
│   ├── train_kd_base.py
│   ├── train_kd_soft.py
│   ├── eval_kd.py
│   ├── eval_kd_base.py
│   └── eval_kd_soft.py
│
├── Load Models/
│   ├── LoadTeacher.py
│   └── LoadStudent.py
│
├── Results.txt
├── requirements.txt
└── README.md
````

---

## Methodology

### Models

| Role    | Model                                | Parameters | Description                                                  |
| ------- | ------------------------------------ | ---------: | ------------------------------------------------------------ |
| Teacher | `mistralai/Mistral-7B-Instruct-v0.1` |      7.24B | Instruction-tuned causal LM loaded in 4-bit NF4 quantisation |
| Student | `Locutusque/TinyMistral-248M`        |       248M | Non-instruction-tuned compact causal LM                      |

### Compression Ratio

```text
7.24B / 248M = 15.13× compression
```

### Dataset

| Property              | Value                             |
| --------------------- | --------------------------------- |
| Dataset               | `databricks/databricks-dolly-15k` |
| Split                 | 90% train / 10% validation        |
| Seed                  | 42                                |
| Training Samples Used | 10,000                            |
| Evaluation PPL        | Full validation set               |
| Generation Evaluation | 100 validation samples            |
| Max Sequence Length   | 512                               |
| Prompt Format         | Alpaca-style instruction template |

### Prompt Format

```text
### Instruction:
{instruction}

### Input:
{optional_context}

### Response:
{response}
```

---

## Pre-Distillation Fine-Tuning

### Standard SFT

* LoRA rank: `r = 8`
* LoRA alpha: `16`
* Target modules: `q_proj`, `v_proj`
* Label smoothing: `0.0`
* Hard-label cross-entropy

### Soft SFT

* LoRA rank: `r = 4`
* LoRA alpha: `8`
* Target modules: `q_proj`
* Label smoothing: `0.2`
* Reduced adaptation strength for softer student initialisation

Both stages:

* Use cosine LR scheduler
* Train for 1 epoch
* Use effective batch size = 16
* Merge LoRA weights into dense checkpoints before KD

---

## Distillation Pipeline

All three KD conditions use identical distillation hyperparameters.

Only the student initialisation changes.

### Distillation Procedure

```text
1. Student forward pass
2. Teacher forward pass (frozen, no_grad)
3. Shift logits for causal prediction
4. Mask prompt tokens
5. Compute softened distributions
6. Compute forward KL divergence
7. Compute CE loss
8. Combine losses
```

### Distillation Objective

The total loss is:

```math
L_total = (1 - α) · L_CE + α · L_KL
```

Where:

```math
L_KL = τ² · KL(p_T || p_S)
```

### Distillation Hyperparameters

| Hyperparameter        | Value      |
| --------------------- | ---------- |
| Temperature τ         | 4.5        |
| Alpha α               | 0.9        |
| Distillation Type     | Forward KL |
| Epochs                | 7          |
| Learning Rate         | 1e-4       |
| Batch Size            | 2          |
| Gradient Accumulation | 8          |
| Effective Batch Size  | 16         |
| Weight Decay          | 0.05       |
| Scheduler             | Cosine     |
| Teacher Precision     | 4-bit NF4  |
| Student Precision     | float32    |

---

## Experimental Conditions

| Condition     | Description                           |
| ------------- | ------------------------------------- |
| Baseline      | Raw TinyMistral-248M with no training |
| SFT           | Standard supervised fine-tuning only  |
| SFT-Soft      | Label-smoothed softer SFT only        |
| BaseModel KD  | KD directly on raw base model         |
| KD + SFT      | KD after standard SFT                 |
| KD + SFT-Soft | KD after soft label-smoothed SFT      |

---

# Results

## Table 1 — Full Results Across All Conditions

> Lower is better: PPL, KL Divergence, Latency
> Higher is better: ROUGE, BLEU, METEOR, BERTScore, Throughput

| Condition     |   PPL ↓ | KL Div ↓ | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU ↑ | METEOR ↑ | BERTScore ↑ | Latency ↓ |     Speed ↑ |
| ------------- | ------: | -------: | --------: | --------: | --------: | -----: | -------: | ----------: | --------: | ----------: |
| Baseline      | 35.1700 |        — |    0.0770 |    0.0160 |    0.0700 | 0.0100 |   0.0750 |      0.7830 |  2.3300 s | 55.0900 t/s |
| SFT           | 31.5600 |        — |    0.1470 |    0.0530 |    0.1330 | 0.0080 |   0.1090 |      0.8080 |  2.7600 s | 46.3700 t/s |
| SFT-Soft      | 36.4000 |        — |    0.0860 |    0.0220 |    0.0790 | 0.0160 |   0.0740 |      0.7680 |  2.5500 s | 50.3300 t/s |
| BaseModel KD  | 39.0600 |   2.1710 |    0.1800 |    0.0570 |    0.1460 | 0.0130 |   0.1270 |      0.8367 |         — |           — |
| KD + SFT      | 40.3700 |   2.2090 |    0.1890 |    0.0650 |    0.1578 | 0.0130 |   0.1427 |      0.8300 |  0.9000 s | 50.6800 t/s |
| KD + SFT-Soft | 38.6804 |   2.1750 |    0.1770 |    0.0590 |    0.1470 | 0.0130 |   0.1330 |      0.8370 |  0.9400 s | 50.0300 t/s |

---

## Table 2 — Percentage Change Relative to Baseline

| Condition     |     PPL ↓ |  ROUGE-1 ↑ |  ROUGE-2 ↑ |  ROUGE-L ↑ |    BLEU ↑ |  METEOR ↑ | BERTScore ↑ | Latency ↓ |   Speed ↑ |
| ------------- | --------: | ---------: | ---------: | ---------: | --------: | --------: | ----------: | --------: | --------: |
| Baseline      |         — |          — |          — |          — |         — |         — |           — |         — |         — |
| SFT           | -10.2600% |  +91.9100% | +240.6500% |  +91.5200% | -12.5000% | +45.2700% |    +3.1800% | +18.4500% | -15.8200% |
| SFT-Soft      |  +3.5100% |  +12.1400% |  +38.7100% |  +12.9300% | +67.7100% |  -1.0700% |    -1.9200% |  +9.4400% |  -8.6500% |
| BaseModel KD  | +11.0900% | +134.8600% | +270.3200% | +109.9100% | +35.4200% | +69.5100% |    +6.9000% |         — |         — |
| KD + SFT      | +14.8100% | +146.2100% | +318.7100% | +126.7200% | +32.2900% | +90.0100% |    +6.0400% | -61.3700% |  -8.0100% |
| KD + SFT-Soft | +10.0000% | +131.2000% | +281.2900% | +111.0600% | +32.2900% | +76.8300% |    +6.9400% | -59.6600% |  -9.1900% |

---

## Key Findings

### 1. KD dramatically improves generation quality

All distilled models significantly outperform their non-distilled counterparts across:

* ROUGE
* METEOR
* BERTScore
* Semantic instruction-following quality

Despite higher perplexity values.

### 2. Pre-distillation fine-tuning helps

* `KD + SFT` achieves the strongest generation metrics
* `KD + SFT-Soft` achieves the best PPL among KD variants
* Softer initialisation improves calibration and semantic alignment

### 3. Perplexity and generation quality diverge

```text
Lower perplexity ≠ better instruction-following quality
```

SFT achieves the best PPL but is outperformed by all KD models on generation metrics.

### 4. Distillation achieves substantial efficiency gains

| Metric              | Result        |
| ------------------- | ------------- |
| Compression Ratio   | 15.13×        |
| KD + SFT Latency    | 0.90 s/prompt |
| Baseline Latency    | 2.33 s/prompt |
| Approximate Speedup | ~3×           |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Khagendra-12/KnowledgeDistillation-under-data-constraints.git
cd KnowledgeDistillation-under-data-constraints
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Baseline Evaluation

```bash
cd Baseline
python baseline_eval.py
```

### Standard SFT

```bash
cd Baseline
python finetune.py
python merge_lora.py
python eval_sft.py
```

### Soft SFT

```bash
cd Baseline
python finetune_soft.py
python merge_lora_soft.py
python eval_sft_soft.py
```

### Distillation — Raw Base Model

```bash
cd Distillation_Pipeline
python train_kd_base.py
python eval_kd_base.py
```

### Distillation — KD + SFT

```bash
cd Distillation_Pipeline
python train_kd.py
python eval_kd.py
```

### Distillation — KD + SFT-Soft

```bash
cd Distillation_Pipeline
python train_kd_soft.py
python eval_kd_soft.py
```

---

## Limitations

* Single teacher-student architecture family
* Single dataset evaluation
* No downstream benchmark evaluation
* Perplexity evaluation mismatch between training and evaluation
* No reverse-KL comparison

---

## Future Work

* Cross-architecture distillation
* Lower data-budget studies
* Reverse-KL vs Forward-KL comparison
* Benchmark evaluation (MMLU, HellaSwag, WinoGrande)
* Distillation on reasoning and code datasets

---

## Citation

```bibtex
@article{khagendra2026kd,
  title     = {Knowledge Distillation for Non-Instruction-Tuned Compact Language Models Under Data Constraints and Impact of Pre-Distillation Fine-Tuning},
  author    = {Khagendra Saini and Payal Mittal},
  journal   = {IEEE},
  year      = {2026},
  note      = {Manuscript in preparation}
}
```

---

## Acknowledgements

**Thapar Institute of Engineering & Technology**
Department of Computer Science & Engineering
Patiala, India

```
```
