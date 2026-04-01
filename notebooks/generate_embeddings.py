import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel

# check GPU
print("GPU available:", torch.cuda.is_available())

# tokenizer + model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

device = torch.device("cuda")
model.to(device)
model.eval()

# === PATH CORRETTI ===
DATA_PATH = "/work/u10767335/data/amazon_model_dataset_10k.csv"
OUTPUT_PATH = "/work/u10767335/embeddings/embeddings_10k.pt"

# load data
df = pd.read_csv(DATA_PATH)
sentences = df["sentence"].tolist()

all_embeddings = []
batch_size = 32

for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]  # CLS token

    all_embeddings.append(emb.cpu())

    if i % 500 == 0:
        print(f"{i}/{len(sentences)}")

all_embeddings = torch.cat(all_embeddings)

print("Final shape:", all_embeddings.shape)

# save embeddings
torch.save(all_embeddings, OUTPUT_PATH)

print("Saved in:", OUTPUT_PATH)