<div align="center">

# Knowledge Distillation for Non-Instruction-Tuned Compact Language Models Under Data Constraints

### Impact of Pre-Distillation Fine-Tuning on Student Model Performance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue?style=flat-square)](https://ieee.org)

**Teacher:** `mistralai/Mistral-7B-Instruct-v0.1` (7.24B params) &nbsp;→&nbsp; **Student:** `Locutusque/TinyMistral-248M` (248M params)

**15.13× Compression** &nbsp;|&nbsp; **~3× Faster Inference** &nbsp;|&nbsp; **~83% BERTScore Retention**

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

> **"Knowledge Distillation for Non-Instruction-Tuned Compact Language Models Under Data Constraints & Impact of Pre-Distillation Fine-Tuning"**[cite: 1]

Large language models (LLMs) achieve strong performance across NLP tasks but are prohibitively expensive to deploy on resource-constrained hardware. **Knowledge Distillation (KD)** offers a principled way to transfer capabilities from a large *teacher* model to a compact *student* model. However, most prior work assumes access to large instruction-following datasets, and the effect of pre-distillation fine-tuning on non-instruction-tuned base models remains underexplored.

This project systematically investigates:

1. Whether KD can be successfully applied to a **non-instruction-tuned compact base model** under constrained data conditions.
2. Whether a **pre-distillation fine-tuning (SFT) stage** improves the quality of the distilled student.
3. Whether the **softness of that pre-FT stage** (standard hard-label SFT vs. label-smoothed SFT) makes a measurable difference.

All experiments use a fixed 10,000-sample budget from the Databricks Dolly-15k dataset to simulate real-world data constraints.

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

```text
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
