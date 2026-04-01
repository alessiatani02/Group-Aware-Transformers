import torch
import pandas as pd

import argparse
import torch.serialization
torch.serialization.add_safe_globals([argparse.Namespace])

from typing import Tuple
from torch import cat, flatten
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens


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

        self.roberta = torch.hub.load("pytorch/fairseq", "roberta.base", weights_only=False)
        # Freeze all BERT layers
        if self.train_bert_layers <= 0:
            self.roberta.eval()
        # Train last `train_bert_layers` BERT layers
        else:
            for param in self.roberta.model.encoder.sentence_encoder.embed_tokens.parameters():
                param.requires_grad = False
            for param in self.roberta.model.encoder.sentence_encoder.embed_positions.parameters():
                param.requires_grad = False
            for layer in self.roberta.model.encoder.sentence_encoder.layers[:-self.train_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self, embeddings, participant=None, item=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method to be overridden by subclasses."""
        prediction, random_loss = None, None
        return prediction, random_loss

    def embed(self, items: pd.DataFrame) -> torch.Tensor:
        """Creates text+hypothesis embeddings for each item using RoBERTa."""
        texts, hypotheses = items.sentence.values, items.hypothesis.values

        # Freeze all BERT layers
        if self.train_bert_layers <= 0:

            import torch
import pandas as pd

import argparse
import torch.serialization
torch.serialization.add_safe_globals([argparse.Namespace])

from typing import Tuple
from torch import cat, flatten
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens


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

        self.roberta = torch.hub.load("pytorch/fairseq", "roberta.base", weights_only=False)
        # Freeze all BERT layers
        if self.train_bert_layers <= 0:
            self.roberta.eval()
        # Train last `train_bert_layers` BERT layers
        else:
            for param in self.roberta.model.encoder.sentence_encoder.embed_tokens.parameters():
                param.requires_grad = False
            for param in self.roberta.model.encoder.sentence_encoder.embed_positions.parameters():
                param.requires_grad = False
            for layer in self.roberta.model.encoder.sentence_encoder.layers[:-self.train_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self, embeddings, participant=None, item=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method to be overridden by subclasses."""
        prediction, random_loss = None, None
        return prediction, random_loss

    def embed(self, items: pd.DataFrame) -> torch.Tensor:
        import numpy as np
        import torch

        all_embeddings = torch.load(r"C:\TESI\embeddings.pt").numpy()

        batch_indices = items.index.values
        batch_embeddings = all_embeddings[batch_indices]

        batch_embeddings = np.expand_dims(batch_embeddings, axis=1)

        return torch.tensor(batch_embeddings, dtype=torch.float32)
