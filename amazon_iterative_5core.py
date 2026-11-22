import json
import gzip
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================


INPUT_FILE = Path("/work/u10767335/data/Cell_Phones_and_Accessories.jsonl.gz")

OUT_DIR = Path("/work/u10767335/data/cellphones_5core_iterative")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_5CORE_JSONL = OUT_DIR / "Cell_Phones_and_Accessories_5core_iterative.jsonl"
OUTPUT_MODEL_CSV = OUT_DIR / "cellphones_5core_model_dataset.csv.gz"

MIN_REVIEWS = 5
CHUNKSIZE = 300_000


MAX_ROWS = None

# ============================================================
# 1. LOAD MINIMAL COLUMNS
# ============================================================

print("Loading minimal columns...")

reader = pd.read_json(
    INPUT_FILE,
    compression="gzip",
    lines=True,
    chunksize=CHUNKSIZE
)

parts = []
n_read = 0

for chunk in reader:
    if MAX_ROWS is not None and n_read >= MAX_ROWS:
        break

    if MAX_ROWS is not None:
        remaining = MAX_ROWS - n_read
        chunk = chunk.iloc[:remaining]

    parts.append(chunk[["user_id", "parent_asin"]].copy())
    n_read += len(chunk)

    print(f"Rows loaded: {n_read:,}")

df_core = pd.concat(parts, ignore_index=True)

print("\nInitial shape:", df_core.shape)

# ============================================================
# 2. ITERATIVE 5-CORE FILTER
# ============================================================

print("\nStarting iterative 5-core filtering...")

iteration = 0

while True:
    iteration += 1
    n_before = len(df_core)

    user_counts = df_core["user_id"].value_counts()
    item_counts = df_core["parent_asin"].value_counts()

    valid_users = set(user_counts[user_counts >= MIN_REVIEWS].index)
    valid_items = set(item_counts[item_counts >= MIN_REVIEWS].index)

    df_core = df_core[
        df_core["user_id"].isin(valid_users)
        & df_core["parent_asin"].isin(valid_items)
    ].copy()

    n_after = len(df_core)

    print(f"Iteration {iteration}: {n_before:,} -> {n_after:,}")

    if n_after == n_before:
        break

print("\nIterative 5-core completed.")

valid_users_final = set(df_core["user_id"].unique())
valid_items_final = set(df_core["parent_asin"].unique())

print("Final reviews:", f"{len(df_core):,}")
print("Final users:", f"{len(valid_users_final):,}")
print("Final products/groups:", f"{len(valid_items_final):,}")

print("Min reviews per user:", df_core["user_id"].value_counts().min())
print("Min reviews per product/group:", df_core["parent_asin"].value_counts().min())

# ============================================================
# 3. WRITE FULL 5-CORE JSONL
# ============================================================

print("\nWriting full 5-core JSONL...")

n_written = 0
rating_counts = Counter()
user_counts_final = Counter()
item_counts_final = Counter()

with gzip.open(INPUT_FILE, "rt", encoding="utf-8") as f_in, \
     open(OUTPUT_5CORE_JSONL, "w", encoding="utf-8") as f_out:

    for line_num, line in enumerate(f_in, start=1):
        if MAX_ROWS is not None and line_num > MAX_ROWS:
            break

        row = json.loads(line)

        user = row["user_id"]
        item = row["parent_asin"]

        if user in valid_users_final and item in valid_items_final:
            f_out.write(json.dumps(row) + "\n")

            n_written += 1
            rating_counts[row["rating"]] += 1
            user_counts_final[user] += 1
            item_counts_final[item] += 1

        if line_num % 1_000_000 == 0:
            print(f"Processed: {line_num:,} | written: {n_written:,}")

print("\nSaved JSONL:")
print(OUTPUT_5CORE_JSONL)

# ============================================================
# 4. FINAL CHECKS
# ============================================================

print("\n================ FINAL 5-CORE CHECKS ================")
print("Final reviews:", f"{n_written:,}")
print("Final users:", f"{len(user_counts_final):,}")
print("Final products/groups:", f"{len(item_counts_final):,}")

print("Min reviews per user:", min(user_counts_final.values()))
print("Min reviews per product/group:", min(item_counts_final.values()))

print("\nReviews per user:")
print(pd.Series(user_counts_final).describe())

print("\nReviews per product/group:")
print(pd.Series(item_counts_final).describe())

rating_5 = pd.Series(rating_counts).sort_index()

print("\nRating distribution:")
print(rating_5)

print("\nRating distribution (%):")
print((rating_5 / rating_5.sum() * 100).round(2))

neg = rating_5.get(1.0, 0) + rating_5.get(2.0, 0)
neu = rating_5.get(3.0, 0)
pos = rating_5.get(4.0, 0) + rating_5.get(5.0, 0)

print("\nSentiment target:")
print("Negative rating <= 2:", f"{neg:,}", f"({round(neg / n_written * 100, 2)}%)")
print("Neutral rating == 3:", f"{neu:,}", f"({round(neu / n_written * 100, 2)}%)")
print("Positive rating >= 4:", f"{pos:,}", f"({round(pos / n_written * 100, 2)}%)")

binary_total = neg + pos
print("\nAfter removing rating = 3:")
print("Binary total:", f"{binary_total:,}")
print("Negative:", f"{neg:,}", f"({round(neg / binary_total * 100, 2)}%)")
print("Positive:", f"{pos:,}", f"({round(pos / binary_total * 100, 2)}%)")

# ============================================================
# 5. CREATE MODEL-READY DATASET
# ============================================================

print("\nCreating model-ready CSV...")

first = True
n_model_rows = 0

for chunk in pd.read_json(
    OUTPUT_5CORE_JSONL,
    lines=True,
    chunksize=CHUNKSIZE
):
    model_chunk = chunk.rename(columns={
        "parent_asin": "participant",
        "asin": "item"
    })

    model_chunk["target"] = np.select(
        [
            model_chunk["rating"] <= 2,
            model_chunk["rating"] == 3,
            model_chunk["rating"] >= 4
        ],
        [0, 1, 2]
    ).astype(int)

    model_chunk = model_chunk[
        [
            "participant",
            "item",
            "user_id",
            "rating",
            "target",
            "text"
        ]
    ]

    model_chunk.to_csv(
        OUTPUT_MODEL_CSV,
        mode="w" if first else "a",
        header=first,
        index=False,
        compression="gzip"
    )

    n_model_rows += len(model_chunk)
    first = False

    if n_model_rows % 1_000_000 < CHUNKSIZE:
        print(f"Model-ready rows written: {n_model_rows:,}")

print("\nSaved model-ready CSV:")
print(OUTPUT_MODEL_CSV)
print("Rows:", f"{n_model_rows:,}")

print("\nDONE.")