import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, PreTrainedTokenizerFast


# ============================================================
# 1. CONFIG
# ============================================================

TRAIN_PATH = Path(
    "/work/u10767335/data/breath/"
    "breath_train_4class_preprocessed512_nodiag.csv.gz"
)

TEST_PATH = Path(
    "/work/u10767335/data/breath/"
    "breath_test_4class_preprocessed512_nodiag.csv.gz"
)

OUT_DIR = Path(
    "/work/u10767335/data/breath/"
    "group_prefix_umberto_preprocessed512_nodiag_lambda_grid_4class_results"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
MODEL_REVISION = "e934c95baeee3f4e613d426d044e52baa82300df"

TEXT_COL = "testo_512"
TARGET_COL = "diagnosi"

# MODEL GROUP
GROUP_COL = "localita"

# SPLIT GROUP
PATIENT_COL = "pedianet_id"

SEED = 42

BATCH_SIZE = 8
EPOCHS = 5
MAX_LENGTH = 512
NUM_WORKERS = 2

USE_CLASS_WEIGHT = True

LR_CLASSIFIER = 1e-3
LR_PREFIX = 1e-3
LR_LAST_LAYER = 2e-5
WEIGHT_DECAY = 1e-4

# Group-prefix configuration
PREFIX_LENGTH = 1
PREFIX_DIM = 64
PREFIX_INSERT_AFTER_LAYER = 11

# IMPORTANT:
# lambda is selected ONLY on the internal validation set.
# The external test set is evaluated only once after selection.
LAMBDA_GRID = [0.001, 0.01, 0.1]

EPSILON = 1e-7

# Same internal validation protocol as fixed-only
INNER_N_SPLITS = 5
INNER_RANDOM_STATE = 42
INNER_FOLD = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print("Device:", DEVICE)


# ============================================================
# 3. TOKENIZER
# ============================================================

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    bos_token="<s>",
    eos_token="</s>",
    sep_token="</s>",
    cls_token="<s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)


# ============================================================
# 4. LOAD SAVED TRAIN / TEST
# ============================================================

train_full = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
test_df = pd.read_csv(TEST_PATH).reset_index(drop=True)

print("\nSaved train shape:", train_full.shape)
print("Saved test shape:", test_df.shape)

required_cols = {
    TEXT_COL,
    TARGET_COL,
    GROUP_COL,
    PATIENT_COL,
}

for name, df in [("train", train_full), ("test", test_df)]:
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing required columns: {sorted(missing)}"
        )

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    if df[TEXT_COL].str.strip().eq("").any():
        raise ValueError(
            f"{name}: empty texts found in {TEXT_COL}"
        )

    if df[GROUP_COL].isna().any():
        raise ValueError(
            f"{name}: missing values found in {GROUP_COL}"
        )

    if df[PATIENT_COL].isna().any():
        raise ValueError(
            f"{name}: missing values found in {PATIENT_COL}"
        )

train_full["saved_row_id"] = np.arange(
    len(train_full),
    dtype=np.int64,
)

test_df["saved_row_id"] = np.arange(
    len(test_df),
    dtype=np.int64,
)


# ============================================================
# 5. CLASS MAPPING
# ============================================================

CLASS_NAMES = sorted(
    train_full[TARGET_COL]
    .astype(str)
    .unique()
    .tolist()
)

NUM_CLASSES = len(CLASS_NAMES)

if NUM_CLASSES != 4:
    raise ValueError(
        f"Expected 4 classes, found {NUM_CLASSES}: "
        f"{CLASS_NAMES}"
    )

CLASS_TO_ID = {
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}

ID_TO_CLASS = {
    idx: class_name
    for class_name, idx in CLASS_TO_ID.items()
}

unknown_test_classes = (
    set(test_df[TARGET_COL].astype(str))
    - set(CLASS_NAMES)
)

if unknown_test_classes:
    raise ValueError(
        "Test contains classes not present in train: "
        f"{sorted(unknown_test_classes)}"
    )

train_full["y_class"] = (
    train_full[TARGET_COL]
    .astype(str)
    .map(CLASS_TO_ID)
    .astype(np.int64)
)

test_df["y_class"] = (
    test_df[TARGET_COL]
    .astype(str)
    .map(CLASS_TO_ID)
    .astype(np.int64)
)

print("\nClass mapping:", CLASS_TO_ID)

print("\nTrain target distribution:")
print(train_full[TARGET_COL].value_counts())

print("\nTest target distribution:")
print(test_df[TARGET_COL].value_counts())


# ============================================================
# 6. GROUP MAPPING
#    Learned from training only.
# ============================================================

TRAIN_GROUP_NAMES = sorted(
    train_full[GROUP_COL]
    .astype(str)
    .unique()
    .tolist()
)

GROUP_TO_ID = {
    group_name: idx
    for idx, group_name in enumerate(TRAIN_GROUP_NAMES)
}

N_GROUPS = len(GROUP_TO_ID)

train_full["group_id"] = (
    train_full[GROUP_COL]
    .astype(str)
    .map(GROUP_TO_ID)
    .astype(np.int64)
)

unknown_test_groups = (
    set(test_df[GROUP_COL].astype(str))
    - set(TRAIN_GROUP_NAMES)
)

if unknown_test_groups:
    raise ValueError(
        "External test contains unseen localities. "
        f"Unseen: {sorted(unknown_test_groups)}"
    )

test_df["group_id"] = (
    test_df[GROUP_COL]
    .astype(str)
    .map(GROUP_TO_ID)
    .astype(np.int64)
)

patient_overlap = (
    set(train_full[PATIENT_COL])
    & set(test_df[PATIENT_COL])
)

print("\nNumber of groups:", N_GROUPS)
print("Patient overlap train/test:", len(patient_overlap))

if patient_overlap:
    raise ValueError(
        "Train/test patient overlap is not zero."
    )


# ============================================================
# 7. INTERNAL PATIENT-DISJOINT TRAIN / VALIDATION SPLIT
# ============================================================

sgkf = StratifiedGroupKFold(
    n_splits=INNER_N_SPLITS,
    shuffle=True,
    random_state=INNER_RANDOM_STATE,
)

all_splits = list(
    sgkf.split(
        train_full,
        y=train_full["y_class"],
        groups=train_full[PATIENT_COL],
    )
)

fit_idx, val_idx = all_splits[INNER_FOLD]

fit_df = (
    train_full
    .iloc[fit_idx]
    .copy()
    .reset_index(drop=True)
)

val_df = (
    train_full
    .iloc[val_idx]
    .copy()
    .reset_index(drop=True)
)

fit_val_patient_overlap = (
    set(fit_df[PATIENT_COL])
    & set(val_df[PATIENT_COL])
)

print("\nInner split random_state:", INNER_RANDOM_STATE)
print("Inner split fold:", INNER_FOLD)
print("Fit shape:", fit_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)
print(
    "Patient overlap fit/validation:",
    len(fit_val_patient_overlap),
)
print("Fit localities:", fit_df[GROUP_COL].nunique())
print(
    "Validation localities:",
    val_df[GROUP_COL].nunique(),
)
print("Test localities:", test_df[GROUP_COL].nunique())

if fit_val_patient_overlap:
    raise ValueError(
        "Fit/validation patient overlap is not zero."
    )

if (
    set(fit_df[GROUP_COL].astype(str))
    != set(TRAIN_GROUP_NAMES)
):
    raise ValueError(
        "Not all localities are represented in inner fit."
    )

if (
    set(val_df[GROUP_COL].astype(str))
    != set(TRAIN_GROUP_NAMES)
):
    raise ValueError(
        "Not all localities are represented in validation."
    )


# ============================================================
# 8. DATASET / DATALOADER
# ============================================================

class ClinicalTextDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return (
            str(row[TEXT_COL]),
            int(row["y_class"]),
            int(row["group_id"]),
            int(row["saved_row_id"]),
        )


def collate_fn(batch):

    encoded = tokenizer(
        [row[0] for row in batch],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return (
        encoded["input_ids"],
        encoded["attention_mask"],

        torch.tensor(
            [row[1] for row in batch],
            dtype=torch.long,
        ),

        torch.tensor(
            [row[2] for row in batch],
            dtype=torch.long,
        ),

        torch.tensor(
            [row[3] for row in batch],
            dtype=torch.long,
        ),
    )


def make_loader(
    dataframe,
    shuffle,
    seed=SEED,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ClinicalTextDataset(dataframe),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=(
            generator
            if shuffle
            else None
        ),
    )


val_loader = make_loader(
    val_df,
    shuffle=False,
)

test_loader = make_loader(
    test_df,
    shuffle=False,
)


# ============================================================
# 9. CLASS WEIGHTS
# ============================================================

def compute_class_weights(labels):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    )

    if np.any(counts == 0):
        raise ValueError(
            "At least one class is absent: "
            f"{counts.tolist()}"
        )

    weights = (
        len(labels)
        /
        (
            NUM_CLASSES
            * counts.astype(np.float64)
        )
    )

    return (
        torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE,
        ),
        counts,
    )


def build_criterion(labels):

    if not USE_CLASS_WEIGHT:
        return nn.CrossEntropyLoss(), None, None

    weights, counts = compute_class_weights(labels)

    print("Class counts:")

    for idx, class_name in ID_TO_CLASS.items():
        print(
            f"  {class_name}: "
            f"{int(counts[idx])}"
        )

    print(
        "Class weights:",
        weights.detach().cpu().numpy(),
    )

    return (
        nn.CrossEntropyLoss(
            weight=weights
        ),
        weights,
        counts,
    )


# ============================================================
# 10. MODEL
# ============================================================

class UmbertoGroupPrefixClassifier(nn.Module):

    def __init__(
        self,
        n_groups,
        prefix_length=1,
        prefix_dim=64,
    ):
        super().__init__()

        self.umberto = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            use_safetensors=True,
            attn_implementation="eager",
        )

        self.hidden_size = (
            self.umberto.config.hidden_size
        )

        self.num_layers = len(
            self.umberto.encoder.layer
        )

        if self.num_layers != 12:
            raise ValueError(
                "Expected 12 transformer layers, "
                f"found {self.num_layers}"
            )

        if (
            PREFIX_INSERT_AFTER_LAYER
            != self.num_layers - 1
        ):
            raise ValueError(
                "This script expects the prefix "
                "immediately before the final layer."
            )

        self.n_groups = n_groups
        self.prefix_length = prefix_length
        self.prefix_dim = prefix_dim

        # Freeze all UmBERTo
        for parameter in self.umberto.parameters():
            parameter.requires_grad = False

        # Same adaptation as fixed-only baseline:
        # final transformer layer is trainable
        for parameter in (
            self.umberto
            .encoder
            .layer[-1]
            .parameters()
        ):
            parameter.requires_grad = True

        # Group-specific low-dimensional prefix
        self.group_prefixes = nn.Parameter(
            torch.zeros(
                n_groups,
                prefix_length,
                prefix_dim,
            )
        )

        nn.init.normal_(
            self.group_prefixes,
            mean=0.0,
            std=0.02,
        )

        # Projection 64 -> 768
        self.prefix_projection = nn.Linear(
            prefix_dim,
            self.hidden_size,
            bias=False,
        )

        nn.init.normal_(
            self.prefix_projection.weight,
            mean=0.0,
            std=0.02,
        )

        # Same classifier as fixed-only
        self.classifier = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                128,
                NUM_CLASSES,
            ),
        )

        # Shared Gaussian prior variance
        self.raw_cov_diag = nn.Parameter(
            torch.zeros(1)
        )

    def train(self, mode=True):

        super().train(mode)

        # frozen part stays in eval mode
        self.umberto.eval()

        # final layer is trainable
        self.umberto.encoder.layer[-1].train(
            mode
        )

        return self

    def centered_prefixes(self):

        return (
            self.group_prefixes
            -
            self.group_prefixes.mean(
                dim=0,
                keepdim=True,
            )
        )

    def projected_prefixes(self):

        return self.prefix_projection(
            self.centered_prefixes()
        )

    def _extended_attention_mask(
        self,
        attention_mask,
        hidden,
    ):

        try:

            result = (
                self.umberto
                ._create_attention_masks(
                    attention_mask=attention_mask,
                    encoder_attention_mask=None,
                    embedding_output=hidden,
                    encoder_hidden_states=None,
                    past_key_values=None,
                )
            )

            if isinstance(result, tuple):
                return result[0]

            return result

        except (AttributeError, TypeError):

            extended = (
                attention_mask[
                    :,
                    None,
                    None,
                    :
                ]
                .to(
                    dtype=hidden.dtype
                )
            )

            return (
                1.0 - extended
            ) * torch.finfo(
                hidden.dtype
            ).min

    @staticmethod
    def _hidden(layer_output):

        if isinstance(
            layer_output,
            tuple,
        ):
            return layer_output[0]

        if hasattr(
            layer_output,
            "last_hidden_state",
        ):
            return (
                layer_output
                .last_hidden_state
            )

        return layer_output

    def encode_first_11(
        self,
        input_ids,
        attention_mask,
    ):

        # no gradients through frozen part
        with torch.no_grad():

            hidden = (
                self.umberto
                .embeddings(
                    input_ids=input_ids
                )
            )

            for layer_idx in range(
                PREFIX_INSERT_AFTER_LAYER
            ):

                extended_mask = (
                    self._extended_attention_mask(
                        attention_mask,
                        hidden,
                    )
                )

                layer_output = (
                    self.umberto
                    .encoder
                    .layer[layer_idx](
                        hidden,
                        attention_mask=extended_mask,
                    )
                )

                hidden = self._hidden(
                    layer_output
                )

        return hidden

    def run_last_layer(
        self,
        hidden,
        attention_mask,
    ):

        extended_mask = (
            self._extended_attention_mask(
                attention_mask,
                hidden,
            )
        )

        layer_output = (
            self.umberto
            .encoder
            .layer[-1](
                hidden,
                attention_mask=extended_mask,
            )
        )

        return self._hidden(
            layer_output
        )

    @staticmethod
    def mean_pool(
        hidden,
        attention_mask,
    ):

        mask = (
            attention_mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        return (
            (hidden * mask)
            .sum(dim=1)
            /
            mask.sum(dim=1)
            .clamp_min(1.0)
        )

    def forward_with_prefix(
        self,
        input_ids,
        attention_mask,
        group_ids_batch,
    ):

        hidden = self.encode_first_11(
            input_ids,
            attention_mask,
        )

        prefix = (
            self.projected_prefixes()[
                group_ids_batch
            ]
        )

        prefix_mask = torch.ones(
            hidden.size(0),
            self.prefix_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        combined_hidden = torch.cat(
            [
                prefix,
                hidden,
            ],
            dim=1,
        )

        combined_mask = torch.cat(
            [
                prefix_mask,
                attention_mask,
            ],
            dim=1,
        )

        last_hidden = self.run_last_layer(
            combined_hidden,
            combined_mask,
        )

        # Remove prefix before pooling
        text_hidden = last_hidden[
            :,
            self.prefix_length:,
            :
        ]

        pooled = self.mean_pool(
            text_hidden,
            attention_mask,
        )

        return self.classifier(
            pooled
        )

    def forward_without_prefix(
        self,
        input_ids,
        attention_mask,
    ):

        # Diagnostic ablation only.
        # This is NOT the separately trained
        # fixed-only baseline.
        hidden = self.encode_first_11(
            input_ids,
            attention_mask,
        )

        last_hidden = self.run_last_layer(
            hidden,
            attention_mask,
        )

        pooled = self.mean_pool(
            last_hidden,
            attention_mask,
        )

        return self.classifier(
            pooled
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        group_ids_batch,
        return_without_prefix=False,
    ):

        logits_with_prefix = (
            self.forward_with_prefix(
                input_ids,
                attention_mask,
                group_ids_batch,
            )
        )

        if not return_without_prefix:
            return logits_with_prefix

        logits_without_prefix = (
            self.forward_without_prefix(
                input_ids,
                attention_mask,
            )
        )

        return (
            logits_with_prefix,
            logits_without_prefix,
        )

    def prefix_penalty(self):

        prefix = self.centered_prefixes()

        var = (
            F.softplus(
                self.raw_cov_diag
            )
            + EPSILON
        )

        mahalanobis = (
            prefix.pow(2)
            / var
        ).mean()

        logdet = (
            torch.log(var)
            .sum()
        )

        return (
            mahalanobis
            + logdet
        )

    def prefix_variance(self):

        return (
            (
                F.softplus(
                    self.raw_cov_diag
                )
                + EPSILON
            )
            .detach()
            .cpu()
            .item()
        )

    def prefix_stats(self):

        prefix = (
            self.centered_prefixes()
            .detach()
            .reshape(
                self.n_groups,
                -1,
            )
        )

        norms = (
            torch.linalg
            .vector_norm(
                prefix,
                dim=1,
            )
        )

        return {
            "prefix_norm_mean":
                norms.mean().cpu().item(),

            "prefix_norm_min":
                norms.min().cpu().item(),

            "prefix_norm_max":
                norms.max().cpu().item(),

            "prefix_value_std":
                prefix.std(
                    unbiased=False
                ).cpu().item(),
        }


# ============================================================
# 11. OPTIMIZER
# ============================================================

def build_optimizer(model):

    prefix_parameters = [
        model.group_prefixes,
        model.raw_cov_diag,
    ]

    prefix_parameters += list(
        model
        .prefix_projection
        .parameters()
    )

    return torch.optim.AdamW(
        [
            {
                "params":
                    model.classifier.parameters(),
                "lr":
                    LR_CLASSIFIER,
            },
            {
                "params":
                    prefix_parameters,
                "lr":
                    LR_PREFIX,
            },
            {
                "params":
                    model
                    .umberto
                    .encoder
                    .layer[-1]
                    .parameters(),
                "lr":
                    LR_LAST_LAYER,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# 12. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    logits,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    probabilities = (
        torch.softmax(
            torch.tensor(
                logits,
                dtype=torch.float32,
            ),
            dim=1,
        )
        .numpy()
    )

    predictions = (
        probabilities
        .argmax(axis=1)
    )

    recalls = recall_score(
        y_true,
        predictions,
        labels=np.arange(
            NUM_CLASSES
        ),
        average=None,
        zero_division=0,
    )

    try:

        macro_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=np.arange(
                NUM_CLASSES
            ),
            multi_class="ovr",
            average="macro",
        )

        weighted_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=np.arange(
                NUM_CLASSES
            ),
            multi_class="ovr",
            average="weighted",
        )

    except ValueError:

        macro_auc = np.nan
        weighted_auc = np.nan

    result = {
        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "macro_f1":
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "weighted_f1":
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            ),

        "macro_auc":
            macro_auc,

        "weighted_auc":
            weighted_auc,

        "confusion_matrix":
            confusion_matrix(
                y_true,
                predictions,
                labels=np.arange(
                    NUM_CLASSES
                ),
            ),

        "logits":
            np.asarray(logits),

        "prob":
            probabilities,

        "pred":
            predictions,

        "y_true":
            y_true,
    }

    for idx, class_name in (
        ID_TO_CLASS.items()
    ):

        result[
            f"recall_{class_name}"
        ] = float(
            recalls[idx]
        )

    return result


# ============================================================
# 13. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    compare_without_prefix=False,
):

    model.eval()

    logits_with_all = []
    logits_without_all = []

    labels_all = []
    rows_all = []
    groups_all = []

    for (
        input_ids,
        attention_mask,
        labels,
        groups,
        saved_row_ids,
    ) in loader:

        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = (
            attention_mask.to(
                DEVICE,
                non_blocking=True,
            )
        )

        groups_device = (
            groups.to(
                DEVICE,
                non_blocking=True,
            )
        )

        if compare_without_prefix:

            (
                logits_with,
                logits_without,
            ) = model(
                input_ids,
                attention_mask,
                groups_device,
                return_without_prefix=True,
            )

            logits_without_all.append(
                logits_without.cpu()
            )

        else:

            logits_with = model(
                input_ids,
                attention_mask,
                groups_device,
            )

        logits_with_all.append(
            logits_with.cpu()
        )

        labels_all.append(
            labels
        )

        rows_all.append(
            saved_row_ids
        )

        groups_all.append(
            groups
        )

    logits_with = (
        torch.cat(
            logits_with_all
        )
        .numpy()
    )

    labels = (
        torch.cat(
            labels_all
        )
        .numpy()
        .astype(np.int64)
    )

    result = calculate_metrics(
        labels,
        logits_with,
    )

    result["saved_row_id"] = (
        torch.cat(
            rows_all
        )
        .numpy()
        .astype(np.int64)
    )

    result["group_id"] = (
        torch.cat(
            groups_all
        )
        .numpy()
        .astype(np.int64)
    )

    if compare_without_prefix:

        logits_without = (
            torch.cat(
                logits_without_all
            )
            .numpy()
        )

        result_without = (
            calculate_metrics(
                labels,
                logits_without,
            )
        )

        result["without_prefix"] = (
            result_without
        )

        result["logit_impact"] = (
            logits_with
            - logits_without
        )

        result["prob_impact"] = (
            result["prob"]
            - result_without["prob"]
        )

    return result


# ============================================================
# 14. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    lambda_random,
):

    model.train()

    total_losses = []
    ce_losses = []
    prefix_losses = []

    for (
        input_ids,
        attention_mask,
        labels,
        groups,
        _,
    ) in loader:

        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = (
            attention_mask.to(
                DEVICE,
                non_blocking=True,
            )
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        groups = groups.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_ids,
            attention_mask,
            groups,
        )

        ce_loss = criterion(
            logits,
            labels,
        )

        prefix_loss = (
            model.prefix_penalty()
        )

        total_loss = (
            ce_loss
            +
            lambda_random
            * prefix_loss
        )

        total_loss.backward()

        optimizer.step()

        total_losses.append(
            total_loss.item()
        )

        ce_losses.append(
            ce_loss.item()
        )

        prefix_losses.append(
            prefix_loss.item()
        )

    return {
        "train_loss":
            float(
                np.mean(
                    total_losses
                )
            ),

        "train_ce_loss":
            float(
                np.mean(
                    ce_losses
                )
            ),

        "train_prefix_loss":
            float(
                np.mean(
                    prefix_losses
                )
            ),
    }


# ============================================================
# 15. LAMBDA GRID
#
# Every lambda:
#   - starts from exactly the same random seed
#   - uses the same fit/validation split
#   - is evaluated ONLY on validation
#
# The external test is NOT touched here.
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "GROUP-PREFIX BREATH 4-CLASS - UmBERTo"
)

print(
    "Lambda selection on INTERNAL VALIDATION only"
)

print(
    "=" * 80
)

print(
    "Lambda grid:",
    LAMBDA_GRID,
)

fit_criterion, fit_weights, fit_counts = (
    build_criterion(
        fit_df[
            "y_class"
        ].to_numpy()
    )
)

all_lambda_history = []
lambda_summary = []

for lambda_random in LAMBDA_GRID:

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"LAMBDA = {lambda_random}"
    )

    print(
        "-" * 80
    )

    # Same initialization for each lambda:
    # fair comparison.
    set_seed(SEED)

    fit_loader = make_loader(
        fit_df,
        shuffle=True,
        seed=SEED,
    )

    model = (
        UmbertoGroupPrefixClassifier(
            n_groups=N_GROUPS,
            prefix_length=PREFIX_LENGTH,
            prefix_dim=PREFIX_DIM,
        )
        .to(DEVICE)
    )

    optimizer = (
        build_optimizer(
            model
        )
    )

    n_trainable = sum(
        parameter.numel()
        for parameter in (
            model.parameters()
        )
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{n_trainable:,}",
    )

    print(
        "Prefix dimension:",
        PREFIX_DIM,
    )

    print(
        "Prefix length:",
        PREFIX_LENGTH,
    )

    print(
        "Last UmBERTo layer trainable: True"
    )

    best_val_ba_for_lambda = (
        -np.inf
    )

    best_epoch_for_lambda = (
        None
    )

    best_val_metrics_for_lambda = (
        None
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_stats = (
            train_one_epoch(
                model,
                fit_loader,
                optimizer,
                fit_criterion,
                lambda_random,
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            compare_without_prefix=False,
        )

        prefix_stats = (
            model.prefix_stats()
        )

        row = {
            "lambda_random":
                lambda_random,

            "epoch":
                epoch,

            **train_stats,

            "prefix_variance":
                model.prefix_variance(),

            **prefix_stats,

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],

            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],

            "val_weighted_f1":
                val_metrics[
                    "weighted_f1"
                ],

            "val_macro_auc":
                val_metrics[
                    "macro_auc"
                ],

            "val_weighted_auc":
                val_metrics[
                    "weighted_auc"
                ],
        }

        all_lambda_history.append(
            row
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={row['train_loss']:.4f} | "
            f"ce={row['train_ce_loss']:.4f} | "
            f"prefix={row['train_prefix_loss']:.4f} | "
            f"var={row['prefix_variance']:.6f} | "
            f"norm={row['prefix_norm_mean']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_bal_acc={row['val_balanced_accuracy']:.4f} | "
            f"val_f1={row['val_macro_f1']:.4f} | "
            f"val_auc={row['val_macro_auc']:.4f}"
        )

        if (
            val_metrics[
                "balanced_accuracy"
            ]
            >
            best_val_ba_for_lambda
        ):

            best_val_ba_for_lambda = (
                val_metrics[
                    "balanced_accuracy"
                ]
            )

            best_epoch_for_lambda = (
                epoch
            )

            best_val_metrics_for_lambda = {
                "accuracy":
                    val_metrics[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    val_metrics[
                        "balanced_accuracy"
                    ],

                "macro_f1":
                    val_metrics[
                        "macro_f1"
                    ],

                "weighted_f1":
                    val_metrics[
                        "weighted_f1"
                    ],

                "macro_auc":
                    val_metrics[
                        "macro_auc"
                    ],

                "weighted_auc":
                    val_metrics[
                        "weighted_auc"
                    ],
            }

    lambda_summary.append(
        {
            "lambda_random":
                lambda_random,

            "best_epoch":
                best_epoch_for_lambda,

            "best_val_accuracy":
                best_val_metrics_for_lambda[
                    "accuracy"
                ],

            "best_val_balanced_accuracy":
                best_val_metrics_for_lambda[
                    "balanced_accuracy"
                ],

            "best_val_macro_f1":
                best_val_metrics_for_lambda[
                    "macro_f1"
                ],

            "best_val_weighted_f1":
                best_val_metrics_for_lambda[
                    "weighted_f1"
                ],

            "best_val_macro_auc":
                best_val_metrics_for_lambda[
                    "macro_auc"
                ],

            "best_val_weighted_auc":
                best_val_metrics_for_lambda[
                    "weighted_auc"
                ],
        }
    )

    # Free memory before next lambda.
    del model
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 16. SELECT BEST LAMBDA
# ============================================================

history_df = pd.DataFrame(
    all_lambda_history
)

history_df.to_csv(
    OUT_DIR
    /
    "lambda_training_history.csv",
    index=False,
)

lambda_summary_df = (
    pd.DataFrame(
        lambda_summary
    )
    .sort_values(
        [
            "best_val_balanced_accuracy",
            "lambda_random",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

lambda_summary_df.to_csv(
    OUT_DIR
    /
    "lambda_grid_summary.csv",
    index=False,
)

best_row = (
    lambda_summary_df
    .iloc[0]
)

BEST_LAMBDA = float(
    best_row[
        "lambda_random"
    ]
)

BEST_EPOCH = int(
    best_row[
        "best_epoch"
    ]
)

BEST_VALIDATION_BALANCED_ACCURACY = (
    float(
        best_row[
            "best_val_balanced_accuracy"
        ]
    )
)

print(
    "\n"
    + "=" * 80
)

print(
    "LAMBDA GRID SUMMARY"
)

print(
    "=" * 80
)

print(
    lambda_summary_df.to_string(
        index=False
    )
)

print(
    "\nSELECTED LAMBDA:",
    BEST_LAMBDA,
)

print(
    "SELECTED EPOCH:",
    BEST_EPOCH,
)

print(
    "BEST VALIDATION BALANCED ACCURACY:",
    BEST_VALIDATION_BALANCED_ACCURACY,
)

# ============================================================
# 17. FINAL RETRAIN ON ALL SAVED TRAIN
#     using selected lambda + selected epoch
# ============================================================

print(
    "\nRetraining final group-prefix model on the full saved "
    f"training set with lambda={BEST_LAMBDA} "
    f"for {BEST_EPOCH} epoch(s)..."
)

set_seed(SEED)

final_train_loader = make_loader(
    train_full,
    shuffle=True,
    seed=SEED,
)

final_model = (
    UmbertoGroupPrefixClassifier(
        n_groups=N_GROUPS,
        prefix_length=PREFIX_LENGTH,
        prefix_dim=PREFIX_DIM,
    )
    .to(DEVICE)
)

final_criterion, full_weights, full_counts = (
    build_criterion(
        train_full[
            "y_class"
        ].to_numpy()
    )
)

final_optimizer = (
    build_optimizer(
        final_model
    )
)

final_train_history = []

for epoch in range(
    1,
    BEST_EPOCH + 1,
):

    train_stats = (
        train_one_epoch(
            final_model,
            final_train_loader,
            final_optimizer,
            final_criterion,
            BEST_LAMBDA,
        )
    )

    row = {
        "epoch":
            epoch,

        "lambda_random":
            BEST_LAMBDA,

        **train_stats,

        "prefix_variance":
            final_model
            .prefix_variance(),

        **final_model
        .prefix_stats(),
    }

    final_train_history.append(
        row
    )

    print(
        f"Final train epoch {epoch:02d} | "
        f"loss={row['train_loss']:.4f} | "
        f"ce={row['train_ce_loss']:.4f} | "
        f"prefix={row['train_prefix_loss']:.4f} | "
        f"norm={row['prefix_norm_mean']:.4f}"
    )

pd.DataFrame(
    final_train_history
).to_csv(
    OUT_DIR
    /
    "final_retrain_history.csv",
    index=False,
)


# ============================================================
# 18. EXTERNAL TEST - EVALUATED ONCE
# ============================================================

final = evaluate(
    final_model,
    test_loader,
    compare_without_prefix=True,
)

without = (
    final[
        "without_prefix"
    ]
)

print(
    "\nFINAL TEST RESULTS - WITH GROUP PREFIX"
)

print(
    "Selected lambda:",
    BEST_LAMBDA,
)

print(
    "Selected epoch:",
    BEST_EPOCH,
)

print(
    "Accuracy:",
    final["accuracy"],
)

print(
    "Balanced accuracy:",
    final[
        "balanced_accuracy"
    ],
)

print(
    "Macro F1:",
    final["macro_f1"],
)

print(
    "Weighted F1:",
    final[
        "weighted_f1"
    ],
)

print(
    "Macro OVR AUC:",
    final[
        "macro_auc"
    ],
)

print(
    "Weighted OVR AUC:",
    final[
        "weighted_auc"
    ],
)

for class_name in CLASS_NAMES:

    print(
        f"Recall {class_name}:",
        final[
            f"recall_{class_name}"
        ],
    )

print(
    "\nConfusion matrix [rows=true, cols=pred]:"
)

print(
    final[
        "confusion_matrix"
    ]
)

print(
    "Class order:",
    CLASS_NAMES,
)

print(
    "\nDIAGNOSTIC - SAME TRAINED MODEL WITHOUT PREFIX"
)

print(
    "Balanced accuracy without prefix:",
    without[
        "balanced_accuracy"
    ],
)

print(
    "Macro F1 without prefix:",
    without[
        "macro_f1"
    ],
)

print(
    "Macro AUC without prefix:",
    without[
        "macro_auc"
    ],
)

print(
    "NOTE: this is only an ablation of the jointly trained "
    "group-prefix model, not the separately trained "
    "fixed-only baseline."
)


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

pred_df = (
    test_df
    .iloc[
        final[
            "saved_row_id"
        ]
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

pred_df[
    "y_true"
] = final[
    "y_true"
]

pred_df[
    "pred_with_prefix_id"
] = final[
    "pred"
]

pred_df[
    "pred_with_prefix"
] = [
    ID_TO_CLASS[
        int(x)
    ]
    for x in final[
        "pred"
    ]
]

pred_df[
    "pred_without_prefix_id"
] = without[
    "pred"
]

pred_df[
    "pred_without_prefix"
] = [
    ID_TO_CLASS[
        int(x)
    ]
    for x in without[
        "pred"
    ]
]

for (
    idx,
    class_name,
) in ID_TO_CLASS.items():

    safe_name = (
        class_name
        .replace(
            " ",
            "_",
        )
        .replace(
            "/",
            "_",
        )
    )

    pred_df[
        f"prob_with_prefix_{safe_name}"
    ] = final[
        "prob"
    ][
        :,
        idx
    ]

    pred_df[
        f"prob_without_prefix_{safe_name}"
    ] = without[
        "prob"
    ][
        :,
        idx
    ]

    pred_df[
        f"logit_impact_{safe_name}"
    ] = final[
        "logit_impact"
    ][
        :,
        idx
    ]

    pred_df[
        f"prob_impact_{safe_name}"
    ] = final[
        "prob_impact"
    ][
        :,
        idx
    ]

pred_df[
    "mean_abs_logit_impact"
] = (
    np.abs(
        final[
            "logit_impact"
        ]
    )
    .mean(
        axis=1
    )
)

pred_df[
    "mean_abs_prob_impact"
] = (
    np.abs(
        final[
            "prob_impact"
        ]
    )
    .mean(
        axis=1
    )
)

pred_df[
    "prediction_changed"
] = (
    final[
        "pred"
    ]
    !=
    without[
        "pred"
    ]
).astype(
    int
)

pred_df.to_csv(
    OUT_DIR
    /
    "test_predictions.csv",
    index=False,
)


# ============================================================
# 20. SAVE PREFIXES + GROUP SUMMARY
# ============================================================

centered_prefixes = (
    final_model
    .centered_prefixes()
    .detach()
    .cpu()
)

projected_prefixes = (
    final_model
    .projected_prefixes()
    .detach()
    .cpu()
)

torch.save(
    centered_prefixes,
    OUT_DIR
    /
    "group_prefixes.pt",
)

torch.save(
    projected_prefixes,
    OUT_DIR
    /
    "projected_group_prefixes.pt",
)

flat_prefix = (
    centered_prefixes
    .reshape(
        N_GROUPS,
        -1,
    )
    .numpy()
)

group_summary = pd.DataFrame(
    {
        GROUP_COL:
            TRAIN_GROUP_NAMES,

        "group_id":
            np.arange(
                N_GROUPS,
                dtype=np.int64,
            ),

        "prefix_l2_norm":
            np.linalg.norm(
                flat_prefix,
                axis=1,
            ),
    }
)

train_counts = (
    train_full
    .groupby(
        "group_id"
    )
    .size()
    .rename(
        "n_train"
    )
    .reset_index()
)

test_counts = (
    test_df
    .groupby(
        "group_id"
    )
    .size()
    .rename(
        "n_test"
    )
    .reset_index()
)

train_props = (
    pd.crosstab(
        train_full[
            "group_id"
        ],
        train_full[
            TARGET_COL
        ],
        normalize="index",
    )
    .rename(
        columns=lambda c:
            f"train_prop_{c}"
    )
    .reset_index()
)

impact_summary = (
    pred_df
    .groupby(
        "group_id"
    )
    .agg(
        mean_abs_logit_impact=(
            "mean_abs_logit_impact",
            "mean",
        ),

        mean_abs_prob_impact=(
            "mean_abs_prob_impact",
            "mean",
        ),

        prediction_change_rate=(
            "prediction_changed",
            "mean",
        ),
    )
    .reset_index()
)

group_summary = (
    group_summary
    .merge(
        train_counts,
        on="group_id",
        how="left",
    )
    .merge(
        test_counts,
        on="group_id",
        how="left",
    )
    .merge(
        train_props,
        on="group_id",
        how="left",
    )
    .merge(
        impact_summary,
        on="group_id",
        how="left",
    )
)

group_summary.to_csv(
    OUT_DIR
    /
    "group_prefix_summary.csv",
    index=False,
)


# ============================================================
# 21. CONFUSION MATRICES
# ============================================================

pd.DataFrame(
    final[
        "confusion_matrix"
    ],
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(
    OUT_DIR
    /
    "test_confusion_matrix_with_prefix.csv"
)

pd.DataFrame(
    without[
        "confusion_matrix"
    ],
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(
    OUT_DIR
    /
    "test_confusion_matrix_without_prefix.csv"
)


# ============================================================
# 22. FINAL METRICS
# ============================================================

prefix_stats = (
    final_model
    .prefix_stats()
)

metrics = {
    "model":
        "umberto_group_prefix_4class",

    "text_col":
        TEXT_COL,

    "target_col":
        TARGET_COL,

    "group_col":
        GROUP_COL,

    "split_group_col":
        PATIENT_COL,

    "selected_lambda":
        BEST_LAMBDA,

    "best_epoch":
        BEST_EPOCH,

    "best_validation_balanced_accuracy":
        BEST_VALIDATION_BALANCED_ACCURACY,

    "prefix_insert_after_layer":
        PREFIX_INSERT_AFTER_LAYER,

    "prefix_dim":
        PREFIX_DIM,

    "prefix_length":
        PREFIX_LENGTH,

    "n_groups":
        N_GROUPS,

    "n_train":
        len(
            train_full
        ),

    "n_test":
        len(
            test_df
        ),

    "accuracy":
        final[
            "accuracy"
        ],

    "balanced_accuracy":
        final[
            "balanced_accuracy"
        ],

    "macro_f1":
        final[
            "macro_f1"
        ],

    "weighted_f1":
        final[
            "weighted_f1"
        ],

    "macro_auc":
        final[
            "macro_auc"
        ],

    "weighted_auc":
        final[
            "weighted_auc"
        ],

    "balanced_accuracy_without_prefix":
        without[
            "balanced_accuracy"
        ],

    "macro_f1_without_prefix":
        without[
            "macro_f1"
        ],

    "macro_auc_without_prefix":
        without[
            "macro_auc"
        ],

    "mean_abs_logit_impact":
        float(
            np.abs(
                final[
                    "logit_impact"
                ]
            )
            .mean()
        ),

    "mean_abs_prob_impact":
        float(
            np.abs(
                final[
                    "prob_impact"
                ]
            )
            .mean()
        ),

    "prediction_change_rate":
        float(
            np.mean(
                final[
                    "pred"
                ]
                !=
                without[
                    "pred"
                ]
            )
        ),

    "prefix_variance":
        final_model
        .prefix_variance(),

    **prefix_stats,

    "batch_size":
        BATCH_SIZE,

    "epochs_max":
        EPOCHS,

    "lr_classifier":
        LR_CLASSIFIER,

    "lr_prefix":
        LR_PREFIX,

    "lr_last_layer":
        LR_LAST_LAYER,

    "weight_decay":
        WEIGHT_DECAY,

    "use_class_weight":
        USE_CLASS_WEIGHT,

    "n_trainable_parameters":
        n_trainable,
}

for class_name in CLASS_NAMES:

    metrics[
        f"recall_{class_name}"
    ] = final[
        f"recall_{class_name}"
    ]

pd.DataFrame(
    [
        metrics
    ]
).to_csv(
    OUT_DIR
    /
    "final_metrics.csv",
    index=False,
)


# ============================================================
# 23. SAVE FINAL TRAINABLE MODEL PARTS
# ============================================================

torch.save(
    {
        "model_name":
            MODEL_NAME,

        "model_revision":
            MODEL_REVISION,

        "class_names":
            CLASS_NAMES,

        "group_to_id":
            GROUP_TO_ID,

        "selected_lambda":
            BEST_LAMBDA,

        "best_epoch":
            BEST_EPOCH,

        "prefix_length":
            PREFIX_LENGTH,

        "prefix_dim":
            PREFIX_DIM,

        "prefix_insert_after_layer":
            PREFIX_INSERT_AFTER_LAYER,

        "group_prefixes":
            final_model
            .group_prefixes
            .detach()
            .cpu(),

        "raw_cov_diag":
            final_model
            .raw_cov_diag
            .detach()
            .cpu(),

        "prefix_projection_state_dict":
            copy.deepcopy(
                final_model
                .prefix_projection
                .state_dict()
            ),

        "classifier_state_dict":
            copy.deepcopy(
                final_model
                .classifier
                .state_dict()
            ),

        "last_layer_state_dict":
            copy.deepcopy(
                final_model
                .umberto
                .encoder
                .layer[-1]
                .state_dict()
            ),
    },
    OUT_DIR
    /
    "best_model.pt",
)


# ============================================================
# 24. CONFIG JSON
# ============================================================

config = {
    "train_path":
        str(
            TRAIN_PATH
        ),

    "test_path":
        str(
            TEST_PATH
        ),

    "out_dir":
        str(
            OUT_DIR
        ),

    "model_name":
        MODEL_NAME,

    "model_revision":
        MODEL_REVISION,

    "text_col":
        TEXT_COL,

    "target_col":
        TARGET_COL,

    "group_col":
        GROUP_COL,

    "patient_col":
        PATIENT_COL,

    "class_names":
        CLASS_NAMES,

    "class_to_id":
        CLASS_TO_ID,

    "group_to_id":
        GROUP_TO_ID,

    "seed":
        SEED,

    "batch_size":
        BATCH_SIZE,

    "epochs_max":
        EPOCHS,

    "max_length":
        MAX_LENGTH,

    "use_class_weight":
        USE_CLASS_WEIGHT,

    "lr_classifier":
        LR_CLASSIFIER,

    "lr_prefix":
        LR_PREFIX,

    "lr_last_layer":
        LR_LAST_LAYER,

    "weight_decay":
        WEIGHT_DECAY,

    "prefix_length":
        PREFIX_LENGTH,

    "prefix_dim":
        PREFIX_DIM,

    "prefix_insert_after_layer":
        PREFIX_INSERT_AFTER_LAYER,

    "lambda_grid":
        LAMBDA_GRID,

    "selected_lambda":
        BEST_LAMBDA,

    "best_epoch":
        BEST_EPOCH,

    "inner_n_splits":
        INNER_N_SPLITS,

    "inner_random_state":
        INNER_RANDOM_STATE,

    "inner_fold":
        INNER_FOLD,

    "selection_metric":
        "validation_balanced_accuracy",

    "external_test_used_for_lambda_selection":
        False,

    "external_test_used_for_epoch_selection":
        False,
}

with open(
    OUT_DIR
    /
    "config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 25. FINAL OUTPUT SUMMARY
# ============================================================

print(
    "\nSaved:"
)

for filename in [
    "lambda_training_history.csv",
    "lambda_grid_summary.csv",
    "final_retrain_history.csv",
    "test_predictions.csv",
    "final_metrics.csv",
    "group_prefix_summary.csv",
    "group_prefixes.pt",
    "projected_group_prefixes.pt",
    "test_confusion_matrix_with_prefix.csv",
    "test_confusion_matrix_without_prefix.csv",
    "best_model.pt",
    "config.json",
]:

    print(
        OUT_DIR
        /
        filename
    )
