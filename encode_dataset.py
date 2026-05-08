import pandas as pd

INPUT_PATH = "/work/u10767335/data/amazon_model_dataset_full.csv"
OUTPUT_PATH = "/work/u10767335/data/amazon_model_dataset_full_encoded.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

print("Encoding participant...")

df["participant"] = df["participant"].astype("category").cat.codes

print("Saving encoded dataset...")
df.to_csv(OUTPUT_PATH, index=False)

print("DONE encoding")
