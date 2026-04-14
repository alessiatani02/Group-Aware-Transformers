import sys
import pandas as pd
import torch

sys.path.append("/work/u10767335/thesis-mixed-effects/paper_code/nli-mixed-models-master")

from nli_mixed_models.modules.nli_base import NaturalLanguageInference

# crea modello dummy (parametri qualsiasi coerenti)
model = NaturalLanguageInference(
    embedding_dim=768,
    n_predictor_layers=1,
    hidden_dim=128,
    output_dim=3,
    n_participants=100,
    n_items=100,
    setting="test",
    use_item_variance=False,
    use_sampling=False,
    n_samples=1,
    train_bert_layers=0,
    device=torch.device("cpu"),
)

# carica dataset
df = pd.read_csv("/work/u10767335/data/amazon_model_dataset_full.csv")

# prendi un piccolo batch
sample = df.iloc[:5]

# test embed
emb = model.embed(sample)

print("Shape:", emb.shape)
