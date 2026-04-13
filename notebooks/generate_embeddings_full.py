import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import os   

print("GPU available:", torch.cuda.is_available())

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

DATA_PATH = "/work/u10767335/data/Books_5core.jsonl"
OUTPUT_DIR = "/work/u10767335/embeddings_chunks/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df_iter = pd.read_json(DATA_PATH, lines=True, chunksize=50000)

batch_size = 32
existing_files = os.listdir(OUTPUT_DIR)
chunk_id = len(existing_files)

print(f"Starting from chunk {chunk_id}")

for i, df_chunk in enumerate(df_iter):

    if i < chunk_id:
        continue

    sentences = df_chunk["text"].dropna().tolist()

    chunk_embeddings = []

    for j in range(0, len(sentences), batch_size):
        batch = sentences[j:j+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        emb = outputs.last_hidden_state[:, 0, :]  # CLS token

        chunk_embeddings.append(emb.cpu())

    chunk_tensor = torch.cat(chunk_embeddings)

    save_path = os.path.join(
        OUTPUT_DIR, f"embeddings_chunk_{chunk_id}.pt"
    )

    torch.save(chunk_tensor, save_path)

    print(f"Saved chunk {chunk_id} - shape: {chunk_tensor.shape}")

    chunk_id += 1

print("DONE")