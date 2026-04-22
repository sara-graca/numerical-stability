# Numerical Stability in Speech Representations

Investigation of how reducing numerical precision affects the geometry of wav2vec2 word-level representations, using the Russian-French interference corpus at https://www.ortolang.fr/market/corpora/ru-fr_interference.

## What it does

Extracts word embeddings from speech recordings using `facebook/wav2vec2-base`, then computes pairwise cosine distances between same-word pairs across four precision levels (float64, float32, float16, int8) and analyses whether intra/inter-speaker distance structure is preserved.

## Pipeline

Managed with DVC. Stages:

```
parse → extract → distances → visualise
```

| Stage | Script | Output |
|---|---|---|
| Parse corpus | `src/parse_corpus.py` | `data/parsed/words.csv` |
| Extract embeddings | `src/extract_embeddings.py` | `data/features/word_embeddings.parquet` |
| Compute distances | `src/compute_distances.py` | `data/distances/` |
| Visualise | `src/visualise.py` | `data/figures/` |

## Configuration

All parameters are in `params.yaml`.

## Findings

Precision reduction introduces noise but does not fundamentally alter the structure 
of the representation space. Intra-speaker distances remain strictly below 
inter-speaker distances across all precision levels (100% ordering preservation). 
Float32 is effectively equivalent to float64. Float16 and int8 introduce deviations 
on the order of 1e-3 but preserve the intra/inter structure completely.
