import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

DISTANCES_FILE = os.path.join("data", "distances", "distances.parquet")
STATS_FILE     = os.path.join("data", "distances", "precision_stats.csv")
OUTPUT_DIR     = os.path.join("data", "figures")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PRECISIONS = params["distances"]["precisions"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

distances_df = pd.read_parquet(DISTANCES_FILE)
stats_df     = pd.read_csv(STATS_FILE)

# enforce consistent ordering
distances_df["precision"] = pd.Categorical(distances_df["precision"], categories=PRECISIONS, ordered=True)
stats_df["precision"]     = pd.Categorical(stats_df["precision"],     categories=PRECISIONS, ordered=True)
stats_df = stats_df.sort_values("precision")

COLORS     = {"intra": "#2196F3", "inter": "#F44336"}
PREC_COLORS = ["#1a1a2e", "#16213e", "#0f3460", "#e94560"]


# Figure 1: KDE distributions of intra vs inter distances, one panel per precision

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
axes = axes.flatten()

for ax, precision in zip(axes, PRECISIONS):
    sub = distances_df[distances_df["precision"] == precision]

    for pair_type, color in COLORS.items():
        vals = sub[sub["pair_type"] == pair_type]["distance"].dropna().values
        vals_sample = np.random.choice(vals, size=min(10000, len(vals)), replace=False)
        if len(vals) < 2:
            continue
        kde  = gaussian_kde(vals_sample, bw_method="scott")
        x = np.linspace(vals_sample.min(), vals_sample.max(), 500)
        ax.plot(x, kde(x), color=color, linewidth=2, label=pair_type)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)

    ax.set_title(precision, fontsize=12, fontweight="bold")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Distribution of intra- and inter-speaker cosine distances by precision", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "kde_distances_by_precision.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: kde_distances_by_precision.png")


# Figure 2: Mean intra vs inter distances across precision levels

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x      = np.arange(len(PRECISIONS))
width  = 0.35
labels = [str(p) for p in stats_df["precision"]]

# left: grouped bar chart of mean intra / inter
ax = axes[0]
ax.bar(x - width / 2, stats_df["mean_intra"], width, label="intra-speaker", color=COLORS["intra"], alpha=0.85)
ax.bar(x + width / 2, stats_df["mean_inter"], width, label="inter-speaker", color=COLORS["inter"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.34, 0.42)
ax.set_ylabel("Mean cosine distance")
ax.set_title("Mean intra- vs inter-speaker distance by precision")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# right: intra/inter ratio across precisions
ax = axes[1]
ax.plot(labels, stats_df["intra_inter_ratio"], marker="o", linewidth=2,
        color="#333333", markerfacecolor="#e94560", markersize=8)
ax.set_ylim(0.8363, 0.8365)
ax.set_ylabel("Intra / inter ratio")
ax.set_title("Intra/inter ratio across precision levels")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mean_distances_by_precision.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: mean_distances_by_precision.png")


# Figure 3: Deviation from float64 reference per precision

# sort so rows are aligned across precisions
distances_df = distances_df.sort_values(
    ["word", "speaker_i", "speaker_j", "sent_id_i", "sent_id_j", "pair_type", "precision"]
).reset_index(drop=True)

ref_distances = distances_df[distances_df["precision"] == "float64"]["distance"].values

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

other_precisions = [p for p in PRECISIONS if p != "float64"]

for ax, precision, color in zip(axes, other_precisions, PREC_COLORS[1:]):
    sub = distances_df[distances_df["precision"] == precision].reset_index(drop=True)
    deviations = sub["distance"].astype(np.float64).values - ref_distances.astype(np.float64)
    pair_types = sub["pair_type"].values

    for pair_type, pt_color in COLORS.items():
        vals = deviations[pair_types == pair_type]
        vals_sample = np.random.choice(vals, size=min(10000, len(vals)), replace=False)
        if len(vals_sample) < 2:
            continue
        kde = gaussian_kde(vals_sample, bw_method="scott")
        x   = np.linspace(vals_sample.min(), vals_sample.max(), 500)
        ax.plot(x, kde(x), color=pt_color, linewidth=2, label=pair_type)
        ax.fill_between(x, kde(x), alpha=0.15, color=pt_color)

    ax.axvline(0, linestyle="--", color="grey", linewidth=1)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_title(f"{precision} − float64", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance deviation")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Deviation of each precision from the float64 reference", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "deviation_from_float64.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: deviation_from_float64.png")
