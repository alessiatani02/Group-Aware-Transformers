from pathlib import Path
import pandas as pd

from preprocess_letters import (
    load_tokenizer,
    TokenCounter,
    learn_templates,
    preprocess_letter,
    DEFAULT_TOKENIZER,
)

# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path(
    "/work/u10767335/data/breath/breath_train_4class.csv.gz"
)

TEST_PATH = Path(
    "/work/u10767335/data/breath/breath_test_4class.csv.gz"
)

TRAIN_OUT = Path(
    "/work/u10767335/data/breath/breath_train_4class_preprocessed512_nodiag.csv.gz"
)

TEST_OUT = Path(
    "/work/u10767335/data/breath/breath_test_4class_preprocessed512_nodiag.csv.gz"
)

TEXT_COL = "testo"
OUTPUT_COL = "testo_512"

MAX_TOKENS = 512
TAIL_FRACTION = 0.5

# Keep the same behaviour as the professor's default preprocessing.
# If later you decide to remove explicit diagnosis sections, set this to True.
DROP_DIAGNOSIS = True


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

if TEXT_COL not in train.columns or TEXT_COL not in test.columns:
    raise KeyError(f"Column '{TEXT_COL}' not found.")

if train[TEXT_COL].isna().any() or test[TEXT_COL].isna().any():
    raise ValueError("Missing texts found in train or test.")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading UmBERTo tokenizer...")
tokenizer = load_tokenizer(DEFAULT_TOKENIZER)
counter = TokenCounter(tokenizer)


# ============================================================
# LEARN TEMPLATES ON TRAIN ONLY
# ============================================================

print("\nLearning header/footer templates from TRAIN ONLY...")

train_texts = train[TEXT_COL].dropna()

sample = train_texts.sample(
    min(15000, len(train_texts)),
    random_state=0,
)

min_count = max(30, int(0.002 * len(sample)))

templates = learn_templates(
    sample.tolist(),
    min_count=min_count,
)

print("Learned templates:", len(templates))
print("Template min_count:", min_count)


# ============================================================
# APPLY SAME PREPROCESSING TO TRAIN AND TEST
# ============================================================

def preprocess_dataframe(df, name):
    print(f"\nPreprocessing {name}...")

    out = df.copy()

    out[OUTPUT_COL] = [
        preprocess_letter(
            text=t,
            counter=counter,
            max_tokens=MAX_TOKENS,
            tail_fraction=TAIL_FRACTION,
            drop_diagnosis=DROP_DIAGNOSIS,
            templates=templates,
        )
        for t in out[TEXT_COL]
    ]

    return out


train_new = preprocess_dataframe(train, "TRAIN")
test_new = preprocess_dataframe(test, "TEST")


# ============================================================
# CHECKS
# ============================================================

def check_dataset(original, processed, name):
    if len(original) != len(processed):
        raise RuntimeError(f"{name}: number of rows changed.")

    if processed[OUTPUT_COL].isna().any():
        raise RuntimeError(
            f"{name}: {processed[OUTPUT_COL].isna().sum()} missing preprocessed texts."
        )

    empty = processed[OUTPUT_COL].fillna("").str.strip().eq("").sum()
    if empty:
        raise RuntimeError(f"{name}: {empty} empty preprocessed texts.")

    token_counts = processed[OUTPUT_COL].map(counter.total)

    print(f"\n{name} checks")
    print("Rows:", len(processed))
    print("Median tokens:", float(token_counts.median()))
    print("Mean tokens:", float(token_counts.mean()))
    print("Max tokens:", int(token_counts.max()))
    print("Texts > 512:", int((token_counts > MAX_TOKENS).sum()))

    if (token_counts > MAX_TOKENS).any():
        raise RuntimeError(f"{name}: at least one text exceeds 512 tokens.")


check_dataset(train, train_new, "TRAIN")
check_dataset(test, test_new, "TEST")


# ============================================================
# SAVE
# ============================================================

train_new.to_csv(
    TRAIN_OUT,
    index=False,
    compression="gzip",
)

test_new.to_csv(
    TEST_OUT,
    index=False,
    compression="gzip",
)

print("\nSaved:")
print(TRAIN_OUT)
print(TEST_OUT)

print("\nDone.")
