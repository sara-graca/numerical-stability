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

All parameters are in `params.yaml`:

```yaml
corpus:
  dir: data/raw/ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only
  metadata: data/raw/ru-fr_interference/2/metadata_RUFR.csv

parse:
  min_speakers: 2
  min_recordings: 2
  noise_tokens: ["sil", "sp", "spn", "unk", "<unk>", ""]

extract:
  model: facebook/wav2vec2-base
  target_sr: 16000
  min_frames: 400
  pooling: mean

distances:
  precisions: ["float64", "float32", "float16", "int8"]
```
