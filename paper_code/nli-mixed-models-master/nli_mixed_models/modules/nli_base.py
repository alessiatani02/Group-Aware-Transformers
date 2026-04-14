import torch
import pandas as pd
import os

from typing import Tuple
from torch.nn import Module


class NaturalLanguageInference(Module):
    """Base class for random effects and random slopes models"""

    def __init__(
        self,
        embedding_dim: int,
        n_predictor_layers: int,
        hidden_dim: int,
        output_dim: int,
        n_participants: int,
        n_items: int,
        setting: str,
        use_item_variance: bool,
        use_sampling: bool,
        n_samples: int,
        train_bert_layers: int,
        device=torch.device("cpu"),
    ):
        super().__init__()

        # parametri modello
        self.embedding_dim = embedding_dim
        self.n_predictor_layers = n_predictor_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_participants = n_participants
        self.n_items = n_items
        self.setting = setting
        self.use_item_variance = use_item_variance
        self.use_sampling = use_sampling
        self.n_samples = n_samples
        self.train_bert_layers = train_bert_layers
        self.device = device

        #disattivo completamente roberta
        self.roberta = None

        self.emb_dir = "/work/u10767335/embeddings_chunks"
        self.chunk_size = 50000
        self.loaded_chunks = {}  # cache

    def forward(
        self, embeddings, participant=None, item=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction, random_loss = None, None
        return prediction, random_loss
    
    def embed(self, items: pd.DataFrame) -> torch.Tensor:
        
        indices = items.index.values

        embeddings = []

        for idx in indices:

            chunk_id = idx // self.chunk_size
            local_idx = idx % self.chunk_size

            # carica chunk se non già in cache
            if chunk_id not in self.loaded_chunks:
                path = os.path.join(
                    self.emb_dir, f"embeddings_chunk_{chunk_id}.pt"
                )
                self.loaded_chunks[chunk_id] = torch.load(path).to(self.device)

            chunk = self.loaded_chunks[chunk_id]

            embeddings.append(chunk[local_idx])

        batch_embeddings = torch.stack(embeddings)

        return batch_embeddings.unsqueeze(1)