## Evaluation Metrics

| Metric | Baseline | SFT | SFT-Soft | KD + SFT | KD + SFT (Soft) |
|---|---:|---:|---:|---:|---:|
| ROUGE-1 | 0.0766 | 0.1470 | 0.0859 | 0.1770 | 0.1771 |
| ROUGE-2 | 0.0155 | 0.0528 | 0.0215 | 0.0642 | 0.0591 |
| ROUGE-L | 0.0696 | 0.1333 | 0.0786 | 0.1435 | 0.1469 |
| BLEU | 0.0096 | 0.0084 | 0.0161 | 0.0088 | 0.0127 |
| METEOR | 0.0751 | 0.1091 | 0.0743 | 0.1293 | 0.1328 |
| BERTScore (F1) | 0.7827 | 0.8076 | 0.7677 | 0.8280 | 0.8370 |
| Perplexity | 35.1653 | 31.5570 | 36.4003 | 38.0067 | 38.6804 |
| Speed (tokens/s) | 55.09 | 46.37 | 50.33 | 48.62 | 50.03 |
| Avg Latency (s/prompt) | 2.33 | 2.76 | 2.55 | 0.88 | 0.94 |

---

## Percentage Change Compared to Baseline

| Metric | Baseline | SFT | SFT-Soft | KD + SFT | KD + SFT (Soft) |
|---|---:|---:|---:|---:|---:|
| ROUGE-1 | -- | ↑ 91.9% | ↑ 12.1% | ↑ 131.1% | ↑ 131.2% |
| ROUGE-2 | -- | ↑ 240.6% | ↑ 38.7% | ↑ 314.2% | ↑ 281.3% |
| ROUGE-L | -- | ↑ 91.5% | ↑ 12.9% | ↑ 106.2% | ↑ 111.1% |
| BLEU | -- | ↓ 12.5% | ↑ 67.7% | ↓ 8.3% | ↑ 32.3% |
| METEOR | -- | ↑ 45.3% | ↓ 1.1% | ↑ 72.2% | ↑ 76.8% |
| BERTScore (F1) | -- | ↑ 3.2% | ↓ 1.9% | ↑ 5.8% | ↑ 6.9% |
| Perplexity | -- | ↓ 10.3% | ↑ 3.5% | ↑ 8.1% | ↑ 10.0% |
| Speed (tokens/s) | -- | ↓ 15.8% | ↓ 8.6% | ↓ 11.7% | ↓ 9.2% |
| Avg Latency (s/prompt) | -- | ↑ 18.5% | ↑ 9.4% | ↓ 62.2% | ↓ 59.7% |

---

## KL Divergence Comparison

| Model | KL Divergence |
|---|---:|
| KD + SFT | 2.1496 |
| KD + SFT (Soft) | 2.1747 |

The KL divergence values for both distilled models are relatively close, indicating stable teacher-student distribution alignment during knowledge distillation.  

- **KD + SFT** achieved a slightly lower KL divergence, suggesting tighter probability distribution matching with the teacher model.
- **KD + SFT (Soft)** produced marginally higher KL divergence but achieved stronger semantic performance metrics such as BERTScore, METEOR, and ROUGE-L, indicating improved contextual and semantic generation quality.
