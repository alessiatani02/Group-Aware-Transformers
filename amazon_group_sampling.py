import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "/work/u10767335/data/cellphones_5core_iterative/cellphones_5core_model_dataset.csv.gz"
)

OUT_DIR = Path("/work/u10767335/data/cellphones_50k")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "cellphones_50k_3class_group_sampling.csv.gz"
OUT_METADATA = OUT_DIR / "cellphones_50k_3class_metadata.csv"

N = 50_000
SEED = 123
MIN_REVIEWS_PER_GROUP = 5

# ============================================================
# LOAD DATA
# ============================================================

print("Loading Cell Phones 5-core dataset...")

df = pd.read_csv(
    DATA_PATH,
    compression="gzip",
    usecols=["participant", "item", "user_id", "rating", "target", "text"]
)

df = df.reset_index(drop=True)
df["global_idx"] = np.arange(len(df), dtype=np.int64)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nTarget distribution full dataset:")
print(df["target"].value_counts().sort_index())
print((df["target"].value_counts(normalize=True).sort_index() * 100).round(2))

print("\nNumber of groups full dataset:")
print(df["participant"].nunique())

# ============================================================
# CLEANING
# ============================================================

print("\nCleaning...")

df = df.dropna(subset=["participant", "item", "target", "text"]).copy()

df["target"] = pd.to_numeric(df["target"], errors="coerce")
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

df = df[df["target"].isin([0, 1, 2])].copy()

print("Shape after cleaning:", df.shape)

# ============================================================
# GROUP-BASED SAMPLING ON 3 CLASSES
# ============================================================

print("\nGroup-based sampling on 3-class dataset...")

group_sizes = df["participant"].value_counts()

eligible_groups = np.array(
    group_sizes[group_sizes >= MIN_REVIEWS_PER_GROUP].index,
    copy=True
)

print("Eligible groups with at least 5 reviews:", len(eligible_groups))

rng = np.random.default_rng(SEED)
rng.shuffle(eligible_groups)

selected_groups = []
current_n = 0

for g in eligible_groups:
    group_size = group_sizes.loc[g]
    selected_groups.append(g)
    current_n += group_size

    if current_n >= N:
        break

df_50k = df[df["participant"].isin(selected_groups)].copy()

# Keep original product id, then factorize participant for models
df_50k["participant_original"] = df_50k["participant"]
df_50k["participant"] = pd.factorize(df_50k["participant"])[0]

# Preserve original order from full 5-core dataset
df_50k = df_50k.sort_values("global_idx").reset_index(drop=True)

print("\nTarget n samples:", N)
print("Actual n samples:", len(df_50k))
print("N groups:", df_50k["participant"].nunique())

print("\nTarget distribution in subset:")
print(df_50k["target"].value_counts().sort_index())
print((df_50k["target"].value_counts(normalize=True).sort_index() * 100).round(2))

group_counts = df_50k["participant"].value_counts()

print("\nReviews per group in subset:")
print(group_counts.describe())

print("\nMin reviews per group:", group_counts.min())
print("Max reviews per group:", group_counts.max())

print("\nGroup size frequency head:")
print(group_counts.value_counts().sort_index().head(20))

print("\nGroup size frequency tail:")
print(group_counts.value_counts().sort_index().tail(20))

# ============================================================
# SAVE
# ============================================================

print("\nSaving subset...")

df_50k.to_csv(
    OUT_CSV,
    index=False,
    compression="gzip"
)

metadata = df_50k[
    [
        "global_idx",
        "participant",
        "participant_original",
        "item",
        "user_id",
        "rating",
        "target"
    ]
].copy()

metadata.to_csv(OUT_METADATA, index=False)

print("Saved 3-class subset:")
print(OUT_CSV)

print("Saved metadata:")
print(OUT_METADATA)

print("\nFinal check:")
print(df_50k.head())

print("\nDONE.")