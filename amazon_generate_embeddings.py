import os
from pathlib import Path

import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "/work/u10767335/data/cellphones_50k/cellphones_50k_3class_group_sampling.csv.gz"
)

OUT_DIR = Path("/work/u10767335/data/cellphones_50k")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EMBEDDINGS = OUT_DIR / "cellphones_50k_embeddings.pt"
OUT_METADATA = OUT_DIR / "cellphones_50k_metadata.csv"

BATCH_SIZE = 8
MAX_LENGTH = 256

# ============================================================
# SETUP
# ============================================================

print("GPU available:", torch.cuda.is_available(), flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

model.to(device)
model.eval()

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading subset...", flush=True)

df = pd.read_csv(DATA_PATH, compression="gzip")

print("Shape:", df.shape, flush=True)
print("Columns:", df.columns.tolist(), flush=True)

df = df.reset_index(drop=True)

print("\nTarget distribution:", flush=True)
print(df["target"].value_counts().sort_index(), flush=True)

print("\nMissing text:", df["text"].isna().sum(), flush=True)

sentences = df["text"].fillna("").astype(str).tolist()

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating RoBERTa CLS embeddings...", flush=True)

all_embeddings = []

for start in range(0, len(sentences), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(sentences))
    batch = sentences[start:end]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]  # CLS token
    all_embeddings.append(emb.cpu())

    if start % 3200 == 0:
        print(f"Processed {end:,}/{len(sentences):,}", flush=True)

embeddings = torch.cat(all_embeddings, dim=0)

print("\nEmbeddings shape:", embeddings.shape, flush=True)

# ============================================================
# SAVE
# ============================================================

torch.save(embeddings, OUT_EMBEDDINGS)

metadata_cols = [
    "global_idx",
    "participant",
    "participant_original",
    "item",
    "user_id",
    "rating",
    "target"
]

metadata = df[metadata_cols].copy()
metadata.to_csv(OUT_METADATA, index=False)

print("\nSaved embeddings:", OUT_EMBEDDINGS, flush=True)
print("Saved metadata:", OUT_METADATA, flush=True)

print("\nDONE.", flush=True)