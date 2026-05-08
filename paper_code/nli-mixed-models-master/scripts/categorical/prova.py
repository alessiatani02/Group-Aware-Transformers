import pandas as pd

# carica dataset
df = pd.read_csv("/work/u10767335/data/amazon_model_dataset_full.csv")

# prendi 50k come nel training
df = df.iloc[:50000]

# conta recensioni per prodotto
counts = df["participant"].value_counts()

print("Min:", counts.min())
print("Mean:", counts.mean())
print("Max:", counts.max())
