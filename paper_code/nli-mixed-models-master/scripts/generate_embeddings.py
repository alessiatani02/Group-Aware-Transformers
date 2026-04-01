import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# ===== LOAD MODEL =====
roberta = torch.hub.load("pytorch/fairseq", "roberta.base")
roberta.eval()

# ===== LOAD DATA =====
df = pd.read_csv(r"C:\TESI\amazon_model_dataset_10k.csv")

embeddings = []

print("Generating embeddings...")

for text in tqdm(df["sentence"]):

    try:
        tokens = roberta.encode(str(text))[:512]

        with torch.no_grad():
            features = roberta.extract_features(tokens.unsqueeze(0))

        emb = features[0, 0, :].numpy()

    except Exception as e:
        print("Errore su frase:", str(text)[:50])
        emb = np.zeros(768)

    embeddings.append(emb)

# ===== SAVE =====
embeddings = np.array(embeddings)

np.save(r"C:\TESI\embeddings_10k.npy", embeddings)

print("DONE. Shape:", embeddings.shape)