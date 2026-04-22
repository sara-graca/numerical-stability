import os
import time
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

INPUT_FILE  = os.path.join("data", "features", "word_embeddings.parquet")
OUTPUT_FILE = os.path.join("data", "distances", "distances.parquet")
STATS_FILE  = os.path.join("data", "distances", "precision_stats.csv")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PRECISIONS = params["distances"]["precisions"]


# quantisation

def quantise_int8(matrix: np.ndarray) -> np.ndarray:
    mins  = matrix.min(axis=1, keepdims=True)
    maxs  = matrix.max(axis=1, keepdims=True)
    scale = (maxs - mins) / 255.0
    scale = np.where(scale == 0, 1.0, scale)
    q     = np.round((matrix - mins) / scale).astype(np.uint8)
    return q.astype(np.float64) * scale + mins


def cast_embeddings(matrix: np.ndarray, precision: str) -> np.ndarray:
    if precision == "float64":
        return matrix.astype(np.float64)
    elif precision == "float32":
        return matrix.astype(np.float32)
    elif precision == "float16":
        return matrix.astype(np.float16)
    elif precision == "int8":
        return quantise_int8(matrix.astype(np.float64))
    else:
        raise ValueError(f"Unknown precision: {precision}")


def pairwise_cosine_distances(vecs: np.ndarray) -> np.ndarray:
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.where(norms == 0, 1.0, norms)
    sim    = normed @ normed.T
    dist   = 1.0 - sim
    i_idx, j_idx = np.triu_indices(len(vecs), k=1)
    return dist[i_idx, j_idx], i_idx, j_idx

# load data

df          = pd.read_parquet(INPUT_FILE)
emb_cols    = [c for c in df.columns if c.startswith("dim_")]
base_matrix = df[emb_cols].values.astype(np.float64)

# compute distances per precision

all_rows  = []
stat_rows = []

for precision in PRECISIONS:
    print(f"\nProcessing {precision}...")

    t0     = time.perf_counter()
    matrix = cast_embeddings(base_matrix, precision)
    t_cast = time.perf_counter() - t0

    intra_distances = []
    inter_distances = []

    t1 = time.perf_counter()

    for word, word_df in tqdm(df.groupby("word"), desc=precision, unit="word"):
        speakers = word_df["speaker_id"].values
        sent_ids = word_df["sent_id"].values
        positions = df.index.get_indexer(word_df.index)
        vecs = matrix[positions]

        distances, i_idx, j_idx = pairwise_cosine_distances(vecs)

        same_speaker = speakers[i_idx] == speakers[j_idx]
        pair_types = np.where(same_speaker, "intra", "inter")

        intra_distances.extend(distances[same_speaker].tolist())
        inter_distances.extend(distances[~same_speaker].tolist())

        for k in range(len(distances)):
            all_rows.append({
                "precision":  precision,
                "word":       word,
                "speaker_i":  speakers[i_idx[k]],
                "speaker_j":  speakers[j_idx[k]],
                "sent_id_i":  sent_ids[i_idx[k]],
                "sent_id_j":  sent_ids[j_idx[k]],
                "pair_type":  pair_types[k],
                "distance":   distances[k],
            })

    t_dist = time.perf_counter() - t1

    bytes_per_val = {"float64": 8, "float32": 4, "float16": 2, "int8": 1}
    disk_bytes = len(df) * len(emb_cols) * bytes_per_val[precision]

    mean_intra = float(np.nanmean(intra_distances)) if intra_distances else np.nan
    mean_inter = float(np.nanmean(inter_distances)) if inter_distances else np.nan
    ratio = mean_intra / mean_inter if mean_inter else np.nan

    stat_rows.append({
        "precision":         precision,
        "mean_intra":        mean_intra,
        "mean_inter":        mean_inter,
        "intra_inter_ratio": ratio,
        "n_intra_pairs":     len(intra_distances),
        "n_inter_pairs":     len(inter_distances),
        "cast_time_s":       round(t_cast, 4),
        "dist_time_s":       round(t_dist, 4),
        "disk_bytes":        disk_bytes,
        "disk_mb":           round(disk_bytes / 1e6, 3),
    })

    print(f"  mean intra: {mean_intra:.4f}  mean inter: {mean_inter:.4f}  ratio: {ratio:.4f}")
    print(f"  distance computation: {t_dist:.2f}s  estimated disk: {disk_bytes/1e6:.1f} MB")


# save

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

pd.DataFrame(all_rows).to_parquet(OUTPUT_FILE, index=False)
stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(STATS_FILE, index=False)

print(f"\nSaved distances: {OUTPUT_FILE}")
print(f"Saved statistics: {STATS_FILE}")
pd.set_option("display.float_format", lambda x: f"{x:.10f}")
print(stats_df.to_string(index=False))

df = pd.read_parquet("data/distances/distances.parquet")
f64 = df[df["precision"] == "float64"]["distance"].values
f32 = df[df["precision"] == "float32"]["distance"].values
print(np.max(np.abs(f64 - f32)))