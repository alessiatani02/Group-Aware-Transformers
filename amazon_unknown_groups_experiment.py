import copy
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
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel


# ============================================================
# 1. CONFIG
# ============================================================

META_PATH = Path(
    "/work/u10767335/data/cellphones_50k/"
    "cellphones_50k_3class_group_sampling.csv.gz"
)

OUT_DIR_BASE = Path(
    "/work/u10767335/data/cellphones_50k/"
    "unknown_groups_experiment"
)

OUT_DIR_BASE.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_NAME = "roberta-base"

TEXT_COL = "text"
TARGET_COL = "target"
GROUP_COL = "participant_original"

SEED = 123

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

MAX_LENGTH = 128
NUM_WORKERS = 2

USE_CLASS_WEIGHT = True

# ------------------------------------------------------------
# BASELINE MODEL CONFIGURATION
# ------------------------------------------------------------

LAMBDA_RANDOM = 0.001

PREFIX_LENGTH = 1
PREFIX_DIM = 768

# Insert prefix immediately before final RoBERTa layer.
PREFIX_INSERT_AFTER_LAYER = 11

TRAIN_LAST_LAYER = False

# ------------------------------------------------------------
# UNKNOWN-GROUP EXPERIMENT
# ------------------------------------------------------------

# Fraction of complete products held out from training.
UNKNOWN_GROUP_FRACTION = 0.10

# Among reviews belonging to KNOWN products, reserve 20%
# for validation / model selection.
KNOWN_VALIDATION_FRACTION = 0.20

# Compare:
#
# 0.00 = standard training
# 0.10 = 10% of training reviews use zero prefix
#
UNKNOWN_PREFIX_RATES = [
    0.00,
    0.10,
]

EPSILON = 1e-7

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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
# 3. LOAD DATA
# ============================================================

meta = pd.read_csv(META_PATH)

print("\nMetadata shape:", meta.shape)
print("Columns:", meta.columns.tolist())

for required_col in [
    TEXT_COL,
    TARGET_COL,
    GROUP_COL,
]:

    if required_col not in meta.columns:

        raise ValueError(
            f"Missing column: {required_col}"
        )


# ============================================================
# 4. BINARY TARGET
# ============================================================

# Original:
# 0 = negative
# 1 = neutral
# 2 = positive
#
# Keep only negative and positive.

meta = meta[
    meta[TARGET_COL].isin([0, 2])
].copy()

meta[TEXT_COL] = (
    meta[TEXT_COL]
    .fillna("")
    .astype(str)
)

meta = meta[
    meta[TEXT_COL]
    .str.strip()
    .ne("")
].reset_index(drop=True)

meta["y_binary"] = (
    meta[TARGET_COL] == 2
).astype(np.float32)


# ============================================================
# 5. ORIGINAL GROUP INFORMATION
# ============================================================

all_groups = (
    meta[GROUP_COL]
    .drop_duplicates()
    .to_numpy()
)

print("\nAfter filtering:", meta.shape)

print("\nTarget distribution:")
print(
    meta["y_binary"]
    .value_counts()
    .sort_index()
)

print("\nTotal number of groups:", len(all_groups))

print("\nReviews per group:")
print(
    meta.groupby(GROUP_COL)
    .size()
    .describe()
)


# ============================================================
# 6. SPLIT COMPLETE GROUPS:
#    KNOWN vs UNKNOWN
# ============================================================

# IMPORTANT:
#
# We split GROUP IDs, not individual reviews.
#
# Therefore every review belonging to an unknown product
# is completely absent from training and validation.

known_groups, unknown_groups = train_test_split(
    all_groups,
    test_size=UNKNOWN_GROUP_FRACTION,
    random_state=SEED,
)

known_groups = set(known_groups)
unknown_groups = set(unknown_groups)

known_mask = (
    meta[GROUP_COL]
    .isin(known_groups)
)

unknown_mask = (
    meta[GROUP_COL]
    .isin(unknown_groups)
)

known_indices_all = np.where(
    known_mask.to_numpy()
)[0]

unknown_test_idx = np.where(
    unknown_mask.to_numpy()
)[0]


# ============================================================
# 7. SPLIT KNOWN REVIEWS:
#    TRAIN vs VALIDATION
# ============================================================

known_y = (
    meta.iloc[known_indices_all][
        "y_binary"
    ]
    .to_numpy()
)

train_idx, validation_idx = train_test_split(
    known_indices_all,
    test_size=KNOWN_VALIDATION_FRACTION,
    random_state=SEED,
    stratify=known_y,
)


# ============================================================
# 8. CREATE GROUP IDs ONLY FOR KNOWN PRODUCTS
# ============================================================

# Prefix parameters are learned only for known groups.
#
# Unknown groups DO NOT get a trainable group prefix.

known_group_list = sorted(
    list(known_groups)
)

group_to_id = {
    group: idx
    for idx, group
    in enumerate(known_group_list)
}

n_known_groups = len(
    known_group_list
)

# -1 means unknown group.
meta["known_group_id"] = (
    meta[GROUP_COL]
    .map(group_to_id)
    .fillna(-1)
    .astype(np.int64)
)


texts = meta[TEXT_COL].tolist()

y = meta["y_binary"].to_numpy(
    dtype=np.float32
)

group_ids = (
    meta["known_group_id"]
    .to_numpy(
        dtype=np.int64
    )
)


# ============================================================
# 9. PRINT SPLIT DIAGNOSTICS
# ============================================================

print("\n" + "=" * 80)
print("KNOWN / UNKNOWN GROUP SPLIT")
print("=" * 80)

print(
    "Known groups:",
    len(known_groups)
)

print(
    "Unknown groups:",
    len(unknown_groups)
)

print(
    "Training reviews:",
    len(train_idx)
)

print(
    "Known validation reviews:",
    len(validation_idx)
)

print(
    "Unknown test reviews:",
    len(unknown_test_idx)
)


def print_subset_distribution(
    name,
    indices,
):

    subset = meta.iloc[indices]

    print(f"\n{name}")

    print(
        "Number of reviews:",
        len(subset)
    )

    print(
        "Number of groups:",
        subset[GROUP_COL].nunique()
    )

    print("Class distribution:")

    print(
        subset["y_binary"]
        .value_counts()
        .sort_index()
    )

    print("Class proportions:")

    print(
        subset["y_binary"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    group_sizes = (
        subset.groupby(GROUP_COL)
        .size()
    )

    print("Reviews per group:")

    print(
        group_sizes.describe()
    )


print_subset_distribution(
    "TRAIN",
    train_idx,
)

print_subset_distribution(
    "KNOWN VALIDATION",
    validation_idx,
)

print_subset_distribution(
    "UNKNOWN TEST",
    unknown_test_idx,
)


# ============================================================
# 10. CHECK NO GROUP LEAKAGE
# ============================================================

train_groups_check = set(
    meta.iloc[train_idx][GROUP_COL]
)

validation_groups_check = set(
    meta.iloc[validation_idx][GROUP_COL]
)

unknown_groups_check = set(
    meta.iloc[unknown_test_idx][GROUP_COL]
)

assert (
    train_groups_check
    .intersection(
        unknown_groups_check
    )
    == set()
)

assert (
    validation_groups_check
    .intersection(
        unknown_groups_check
    )
    == set()
)

print(
    "\nLeakage check passed: "
    "unknown products never occur in train/validation."
)


# ============================================================
# 11. SAVE SPLIT INFORMATION
# ============================================================

split_df = pd.DataFrame({
    GROUP_COL: all_groups
})

split_df["split"] = (
    split_df[GROUP_COL]
    .apply(
        lambda g:
        "unknown_test"
        if g in unknown_groups
        else "known"
    )
)

split_df.to_csv(
    OUT_DIR_BASE
    /
    "group_split.csv",
    index=False,
)

meta_split = meta[
    [
        GROUP_COL,
        "known_group_id",
        "y_binary",
    ]
].copy()

meta_split["review_split"] = "unused"

meta_split.loc[
    train_idx,
    "review_split"
] = "train"

meta_split.loc[
    validation_idx,
    "review_split"
] = "known_validation"

meta_split.loc[
    unknown_test_idx,
    "review_split"
] = "unknown_test"

meta_split.to_csv(
    OUT_DIR_BASE
    /
    "review_split.csv",
    index=False,
)


# ============================================================
# 12. TOKENIZER + DATASET
# ============================================================

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )
)


class ReviewDataset(Dataset):

    def __init__(
        self,
        subset_indices,
    ):

        self.indices = np.asarray(
            subset_indices
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        position,
    ):

        idx = int(
            self.indices[position]
        )

        return (
            texts[idx],
            y[idx],
            group_ids[idx],
            idx,
        )


def collate_fn(batch):

    batch_texts = [
        row[0]
        for row in batch
    ]

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return (
        encoded["input_ids"],
        encoded["attention_mask"],

        torch.tensor(
            [
                row[1]
                for row in batch
            ],
            dtype=torch.float32,
        ),

        torch.tensor(
            [
                row[2]
                for row in batch
            ],
            dtype=torch.long,
        ),

        torch.tensor(
            [
                row[3]
                for row in batch
            ],
            dtype=torch.long,
        ),
    )


train_loader = DataLoader(
    ReviewDataset(
        train_idx
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

validation_loader = DataLoader(
    ReviewDataset(
        validation_idx
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

unknown_test_loader = DataLoader(
    ReviewDataset(
        unknown_test_idx
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# ============================================================
# 13. MODEL
# ============================================================

class PrefixClassifier(nn.Module):

    def __init__(
        self,
        n_groups,
        prefix_length=1,
        prefix_dim=768,
        prefix_insert_after_layer=11,
        train_last_layer=False,
    ):

        super().__init__()

        # ----------------------------------------------------
        # RoBERTa
        # ----------------------------------------------------

        try:

            self.roberta = (
                RobertaModel
                .from_pretrained(
                    MODEL_NAME,
                    add_pooling_layer=False,
                    attn_implementation="eager",
                )
            )

        except TypeError:

            self.roberta = (
                RobertaModel
                .from_pretrained(
                    MODEL_NAME,
                    add_pooling_layer=False,
                )
            )

        self.hidden_size = (
            self.roberta
            .config
            .hidden_size
        )

        self.num_layers = len(
            self.roberta
            .encoder
            .layer
        )

        self.n_groups = n_groups
        self.prefix_length = prefix_length
        self.prefix_dim = prefix_dim

        self.prefix_insert_after_layer = (
            prefix_insert_after_layer
        )

        self.train_last_layer = (
            train_last_layer
        )

        # ----------------------------------------------------
        # Freeze RoBERTa
        # ----------------------------------------------------

        for parameter in (
            self.roberta.parameters()
        ):

            parameter.requires_grad = False

        if self.train_last_layer:

            for parameter in (
                self.roberta
                .encoder
                .layer[-1]
                .parameters()
            ):

                parameter.requires_grad = True

        # ----------------------------------------------------
        # Learned prefixes ONLY for known groups
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        if (
            prefix_dim
            ==
            self.hidden_size
        ):

            self.prefix_projection = (
                nn.Identity()
            )

        else:

            self.prefix_projection = (
                nn.Linear(
                    prefix_dim,
                    self.hidden_size,
                    bias=False,
                )
            )

            nn.init.normal_(
                self.prefix_projection.weight,
                mean=0.0,
                std=0.02,
            )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                self.hidden_size,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                1,
            ),
        )

        # ----------------------------------------------------
        # Gaussian prior variance
        # ----------------------------------------------------

        self.raw_cov_diag = nn.Parameter(
            torch.zeros(1)
        )

    # ========================================================
    # TRAIN MODE
    # ========================================================

    def train(
        self,
        mode=True,
    ):

        super().train(mode)

        # Frozen RoBERTa remains in eval mode.
        self.roberta.eval()

        if self.train_last_layer:

            self.roberta.encoder.layer[
                -1
            ].train(mode)

        return self

    # ========================================================
    # CENTER PREFIXES
    # ========================================================

    def centered_prefixes(self):

        return (
            self.group_prefixes
            -
            self.group_prefixes.mean(
                dim=0,
                keepdim=True,
            )
        )

    # ========================================================
    # PROJECT PREFIXES
    # ========================================================

    def projected_prefixes(self):

        prefix = (
            self.centered_prefixes()
        )

        return (
            self.prefix_projection(
                prefix
            )
        )

    # ========================================================
    # RUN TRANSFORMER LAYERS
    # ========================================================

    def run_layers(
        self,
        input_ids,
        attention_mask,
        start_layer,
        end_layer,
        hidden=None,
    ):

        if hidden is None:

            hidden = (
                self.roberta.embeddings(
                    input_ids=input_ids
                )
            )

        current_mask = (
            attention_mask
        )

        for layer_idx in range(
            start_layer,
            end_layer,
        ):

            layer = (
                self.roberta
                .encoder
                .layer[layer_idx]
            )

            extended_mask, _ = (
                self.roberta
                ._create_attention_masks(
                    attention_mask=current_mask,
                    encoder_attention_mask=None,
                    embedding_output=hidden,
                    encoder_hidden_states=None,
                    past_key_values=None,
                )
            )

            hidden = layer(
                hidden,
                attention_mask=extended_mask,
            )

        return hidden

    # ========================================================
    # MEAN POOL
    # ========================================================

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
            (hidden * mask).sum(1)
            /
            mask.sum(1)
            .clamp_min(1.0)
        )

    # ========================================================
    # GET PREFIX
    # ========================================================

    def get_prefix(
        self,
        group_ids_batch,
        force_zero_mask=None,
        all_zero=False,
    ):

        """
        Return one prefix per document.

        KNOWN document:
            learned group-specific prefix.

        force_zero_mask=True:
            replace that document's prefix with zero.

        all_zero=True:
            all documents receive zero prefix.
            Used for completely unknown groups.

        IMPORTANT:
        The zero prefix TOKEN is still inserted into
        the Transformer sequence.
        """

        batch_size = (
            group_ids_batch.size(0)
        )

        if all_zero:

            return torch.zeros(
                batch_size,
                self.prefix_length,
                self.hidden_size,
                dtype=self.group_prefixes.dtype,
                device=group_ids_batch.device,
            )

        # All IDs must correspond to known groups here.
        if torch.any(
            group_ids_batch < 0
        ):

            raise ValueError(
                "Unknown group ID passed without "
                "all_zero=True."
            )

        prefix = (
            self.projected_prefixes()[
                group_ids_batch
            ]
        )

        if force_zero_mask is not None:

            force_zero_mask = (
                force_zero_mask
                .bool()
                .view(
                    -1,
                    1,
                    1,
                )
            )

            prefix = torch.where(
                force_zero_mask,
                torch.zeros_like(prefix),
                prefix,
            )

        return prefix

    # ========================================================
    # FORWARD WITH EXPLICIT PREFIX
    # ========================================================

    def forward(
        self,
        input_ids,
        attention_mask,
        group_ids_batch,
        force_zero_mask=None,
        all_zero=False,
    ):

        # ----------------------------------------------------
        # 1. Run layers before prefix insertion
        # ----------------------------------------------------

        hidden = self.run_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_layer=0,
            end_layer=(
                self.prefix_insert_after_layer
            ),
        )

        # ----------------------------------------------------
        # 2. Obtain prefix
        # ----------------------------------------------------

        prefix = self.get_prefix(
            group_ids_batch=group_ids_batch,
            force_zero_mask=force_zero_mask,
            all_zero=all_zero,
        )

        # ----------------------------------------------------
        # 3. Prefix attention mask
        # ----------------------------------------------------

        prefix_mask = torch.ones(
            hidden.size(0),
            self.prefix_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # ----------------------------------------------------
        # 4. Concatenate prefix + text
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 5. Remaining Transformer layers
        # ----------------------------------------------------

        last_hidden = self.run_layers(
            input_ids=input_ids,
            attention_mask=combined_mask,
            start_layer=(
                self.prefix_insert_after_layer
            ),
            end_layer=self.num_layers,
            hidden=combined_hidden,
        )

        # ----------------------------------------------------
        # 6. Remove prefix before pooling
        # ----------------------------------------------------

        text_hidden = last_hidden[
            :,
            self.prefix_length:,
            :,
        ]

        pooled = self.mean_pool(
            text_hidden,
            attention_mask,
        )

        logits = (
            self.classifier(
                pooled
            )
            .squeeze(1)
        )

        return logits

    # ========================================================
    # PREFIX PENALTY
    # ========================================================

    def prefix_penalty(self):

        prefix = (
            self.centered_prefixes()
        )

        var = (
            F.softplus(
                self.raw_cov_diag
            )
            +
            EPSILON
        )

        mahalanobis = (
            prefix.pow(2)
            /
            var
        ).mean()

        logdet = (
            torch.log(var)
            .sum()
        )

        return (
            mahalanobis
            +
            logdet
        )

    # ========================================================
    # PREFIX VARIANCE
    # ========================================================

    def prefix_variance(self):

        var = (
            F.softplus(
                self.raw_cov_diag
            )
            +
            EPSILON
        )

        return (
            var
            .detach()
            .cpu()
            .item()
        )

    # ========================================================
    # PREFIX STATS
    # ========================================================

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
# 14. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    logits,
):

    prob = (
        1.0
        /
        (
            1.0
            +
            np.exp(-logits)
        )
    )

    pred = (
        prob >= 0.5
    ).astype(int)

    try:

        auc = roc_auc_score(
            y_true,
            prob,
        )

    except ValueError:

        auc = np.nan

    return {
        "accuracy":
            accuracy_score(
                y_true,
                pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                pred,
            ),

        "sensitivity":
            recall_score(
                y_true,
                pred,
                pos_label=1,
            ),

        "specificity":
            recall_score(
                y_true,
                pred,
                pos_label=0,
            ),

        "auc":
            auc,

        "confusion_matrix":
            confusion_matrix(
                y_true,
                pred,
                labels=[0, 1],
            ),

        "logits":
            logits,

        "prob":
            prob,

        "pred":
            pred,

        "y_true":
            y_true,
    }


# ============================================================
# 15. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    use_zero_prefix=False,
):

    model.eval()

    logits_all = []
    labels_all = []
    indices_all = []

    for (
        input_ids,
        attention_mask,
        labels,
        groups,
        original_idx,
    ) in loader:

        input_ids = (
            input_ids.to(
                DEVICE,
                non_blocking=True,
            )
        )

        attention_mask = (
            attention_mask.to(
                DEVICE,
                non_blocking=True,
            )
        )

        groups = groups.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            group_ids_batch=groups,
            all_zero=use_zero_prefix,
        )

        logits_all.append(
            logits.cpu()
        )

        labels_all.append(
            labels
        )

        indices_all.append(
            original_idx
        )

    logits = (
        torch.cat(
            logits_all
        )
        .numpy()
    )

    labels = (
        torch.cat(
            labels_all
        )
        .numpy()
        .astype(int)
    )

    original_indices = (
        torch.cat(
            indices_all
        )
        .numpy()
        .astype(int)
    )

    result = calculate_metrics(
        labels,
        logits,
    )

    result["original_idx"] = (
        original_indices
    )

    return result


# ============================================================
# 16. SAVE / RESTORE TRAINABLE STATE
# ============================================================

def save_trainable_state(
    model,
):

    state = {
        "group_prefixes":
            model.group_prefixes
            .detach()
            .cpu()
            .clone(),

        "raw_cov_diag":
            model.raw_cov_diag
            .detach()
            .cpu()
            .clone(),

        "classifier":
            copy.deepcopy(
                model.classifier
                .state_dict()
            ),

        "prefix_projection":
            copy.deepcopy(
                model.prefix_projection
                .state_dict()
            ),
    }

    if TRAIN_LAST_LAYER:

        state["last_layer"] = (
            copy.deepcopy(
                model.roberta
                .encoder
                .layer[-1]
                .state_dict()
            )
        )

    return state


def load_trainable_state(
    model,
    state,
):

    with torch.no_grad():

        model.group_prefixes.copy_(
            state[
                "group_prefixes"
            ].to(DEVICE)
        )

        model.raw_cov_diag.copy_(
            state[
                "raw_cov_diag"
            ].to(DEVICE)
        )

    model.classifier.load_state_dict(
        state["classifier"]
    )

    model.prefix_projection.load_state_dict(
        state["prefix_projection"]
    )

    if TRAIN_LAST_LAYER:

        model.roberta.encoder.layer[
            -1
        ].load_state_dict(
            state["last_layer"]
        )


# ============================================================
# 17. TRAIN ONE EXPERIMENT
# ============================================================

def train_one_model(
    unknown_prefix_rate,
):

    rate_name = (
        str(unknown_prefix_rate)
        .replace(".", "p")
    )

    experiment_name = (
        f"unknown_prefix_rate_{rate_name}"
    )

    out_dir = (
        OUT_DIR_BASE
        /
        experiment_name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)

    print(
        "EXPERIMENT:",
        experiment_name,
    )

    print(
        "Unknown prefix rate:",
        unknown_prefix_rate,
    )

    print(
        "Known groups:",
        n_known_groups,
    )

    print(
        "Unknown test groups:",
        len(unknown_groups),
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Reset seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = PrefixClassifier(
        n_groups=n_known_groups,
        prefix_length=PREFIX_LENGTH,
        prefix_dim=PREFIX_DIM,
        prefix_insert_after_layer=(
            PREFIX_INSERT_AFTER_LAYER
        ),
        train_last_layer=(
            TRAIN_LAST_LAYER
        ),
    ).to(DEVICE)

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    n_trainable = sum(
        p.numel()
        for p in trainable_parameters
    )

    print(
        "Trainable parameters:",
        f"{n_trainable:,}",
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    if USE_CLASS_WEIGHT:

        n_pos = (
            y[train_idx]
            .sum()
        )

        n_neg = (
            len(train_idx)
            -
            n_pos
        )

        pos_weight_value = float(
            n_neg / n_pos
        )

        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=DEVICE,
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            )
        )

        print(
            "pos_weight:",
            pos_weight_value,
        )

    else:

        pos_weight_value = np.nan

        criterion = (
            nn.BCEWithLogitsLoss()
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_balanced_accuracy = (
        -np.inf
    )

    best_epoch = None
    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_losses = []
        fixed_losses = []
        prefix_losses = []

        n_zero_prefix_documents = 0
        n_training_documents = 0

        for (
            input_ids,
            attention_mask,
            labels,
            groups,
            _,
        ) in train_loader:

            input_ids = (
                input_ids.to(
                    DEVICE,
                    non_blocking=True,
                )
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

            # ------------------------------------------------
            # RANDOM UNKNOWN-PREFIX MASK
            #
            # For example with rate=0.10:
            # each training review independently has a 10%
            # probability of receiving the zero prefix.
            #
            # A new mask is sampled for every batch and epoch.
            # ------------------------------------------------

            if unknown_prefix_rate > 0:

                zero_prefix_mask = (
                    torch.rand(
                        groups.size(0),
                        device=DEVICE,
                    )
                    <
                    unknown_prefix_rate
                )

            else:

                zero_prefix_mask = (
                    torch.zeros(
                        groups.size(0),
                        dtype=torch.bool,
                        device=DEVICE,
                    )
                )

            n_zero_prefix_documents += int(
                zero_prefix_mask.sum().item()
            )

            n_training_documents += int(
                groups.size(0)
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                group_ids_batch=groups,
                force_zero_mask=(
                    zero_prefix_mask
                ),
                all_zero=False,
            )

            # ------------------------------------------------
            # Prediction loss
            # ------------------------------------------------

            fixed_loss = criterion(
                logits,
                labels,
            )

            # ------------------------------------------------
            # Prefix penalty
            # ------------------------------------------------

            prefix_loss = (
                model.prefix_penalty()
            )

            # ------------------------------------------------
            # Total loss
            # ------------------------------------------------

            loss = (
                fixed_loss
                +
                LAMBDA_RANDOM
                *
                prefix_loss
            )

            loss.backward()

            optimizer.step()

            total_losses.append(
                loss.item()
            )

            fixed_losses.append(
                fixed_loss.item()
            )

            prefix_losses.append(
                prefix_loss.item()
            )

        # ----------------------------------------------------
        # Training averages
        # ----------------------------------------------------

        train_loss = float(
            np.mean(
                total_losses
            )
        )

        train_fixed_loss = float(
            np.mean(
                fixed_losses
            )
        )

        train_prefix_loss = float(
            np.mean(
                prefix_losses
            )
        )

        actual_zero_rate = (
            n_zero_prefix_documents
            /
            n_training_documents
        )

        # ----------------------------------------------------
        # KNOWN validation
        #
        # IMPORTANT:
        # Use real learned prefixes here.
        #
        # The unknown test set is NOT used to choose
        # the best epoch.
        # ----------------------------------------------------

        validation_metrics = evaluate(
            model=model,
            loader=validation_loader,
            use_zero_prefix=False,
        )

        stats = (
            model.prefix_stats()
        )

        row = {
            "epoch":
                epoch,

            "unknown_prefix_rate":
                unknown_prefix_rate,

            "actual_zero_prefix_rate":
                actual_zero_rate,

            "train_loss":
                train_loss,

            "train_fixed_loss":
                train_fixed_loss,

            "train_prefix_loss":
                train_prefix_loss,

            "prefix_variance":
                model.prefix_variance(),

            **stats,

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "validation_balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],

            "validation_sensitivity":
                validation_metrics[
                    "sensitivity"
                ],

            "validation_specificity":
                validation_metrics[
                    "specificity"
                ],

            "validation_auc":
                validation_metrics[
                    "auc"
                ],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"zero_rate={actual_zero_rate:.3f} | "
            f"val_acc="
            f"{row['validation_accuracy']:.4f} | "
            f"val_bal_acc="
            f"{row['validation_balanced_accuracy']:.4f} | "
            f"val_auc="
            f"{row['validation_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Select best epoch ONLY using known validation set
        # ----------------------------------------------------

        if (
            validation_metrics[
                "balanced_accuracy"
            ]
            >
            best_balanced_accuracy
        ):

            best_balanced_accuracy = (
                validation_metrics[
                    "balanced_accuracy"
                ]
            )

            best_epoch = epoch

            best_state = (
                save_trainable_state(
                    model
                )
            )

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    load_trainable_state(
        model,
        best_state,
    )

    # ========================================================
    # FINAL KNOWN VALIDATION
    # ========================================================

    final_validation = evaluate(
        model=model,
        loader=validation_loader,
        use_zero_prefix=False,
    )

    # ========================================================
    # FINAL UNKNOWN-GROUP TEST
    #
    # ALL unknown products receive exactly the same
    # ZERO PREFIX TOKEN.
    # ========================================================

    final_unknown = evaluate(
        model=model,
        loader=unknown_test_loader,
        use_zero_prefix=True,
    )

    print("\nBEST MODEL")

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Known validation balanced accuracy:",
        final_validation[
            "balanced_accuracy"
        ],
    )

    print(
        "Known validation AUC:",
        final_validation[
            "auc"
        ],
    )

    print("\nUNKNOWN GROUP TEST")

    print(
        "Accuracy:",
        final_unknown[
            "accuracy"
        ],
    )

    print(
        "Balanced accuracy:",
        final_unknown[
            "balanced_accuracy"
        ],
    )

    print(
        "Sensitivity:",
        final_unknown[
            "sensitivity"
        ],
    )

    print(
        "Specificity:",
        final_unknown[
            "specificity"
        ],
    )

    print(
        "AUC:",
        final_unknown[
            "auc"
        ],
    )

    print(
        "Confusion matrix:"
    )

    print(
        final_unknown[
            "confusion_matrix"
        ]
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        out_dir
        /
        "training_history.csv",
        index=False,
    )

    # ========================================================
    # SAVE UNKNOWN TEST PREDICTIONS
    # ========================================================

    unknown_pred_df = (
        meta
        .iloc[
            final_unknown[
                "original_idx"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    unknown_pred_df[
        "prob_zero_prefix"
    ] = final_unknown[
        "prob"
    ]

    unknown_pred_df[
        "pred_zero_prefix"
    ] = final_unknown[
        "pred"
    ]

    unknown_pred_df[
        "y_true"
    ] = final_unknown[
        "y_true"
    ]

    unknown_pred_df.to_csv(
        out_dir
        /
        "unknown_test_predictions.csv",
        index=False,
    )

    # ========================================================
    # SAVE LEARNED PREFIXES FOR KNOWN GROUPS
    # ========================================================

    centered_prefixes = (
        model
        .centered_prefixes()
        .detach()
        .cpu()
    )

    torch.save(
        centered_prefixes,
        out_dir
        /
        "known_group_prefixes.pt",
    )

    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    stats = (
        model.prefix_stats()
    )

    metrics = {
        "unknown_prefix_rate":
            unknown_prefix_rate,

        "unknown_group_fraction":
            UNKNOWN_GROUP_FRACTION,

        "known_validation_fraction":
            KNOWN_VALIDATION_FRACTION,

        "best_epoch":
            best_epoch,

        "n_known_groups":
            n_known_groups,

        "n_unknown_groups":
            len(unknown_groups),

        "n_train":
            len(train_idx),

        "n_known_validation":
            len(validation_idx),

        "n_unknown_test":
            len(unknown_test_idx),

        "prefix_insert_after_layer":
            PREFIX_INSERT_AFTER_LAYER,

        "prefix_dim":
            PREFIX_DIM,

        "prefix_length":
            PREFIX_LENGTH,

        "lambda_random":
            LAMBDA_RANDOM,

        "known_validation_accuracy":
            final_validation[
                "accuracy"
            ],

        "known_validation_balanced_accuracy":
            final_validation[
                "balanced_accuracy"
            ],

        "known_validation_sensitivity":
            final_validation[
                "sensitivity"
            ],

        "known_validation_specificity":
            final_validation[
                "specificity"
            ],

        "known_validation_auc":
            final_validation[
                "auc"
            ],

        "unknown_accuracy":
            final_unknown[
                "accuracy"
            ],

        "unknown_balanced_accuracy":
            final_unknown[
                "balanced_accuracy"
            ],

        "unknown_sensitivity":
            final_unknown[
                "sensitivity"
            ],

        "unknown_specificity":
            final_unknown[
                "specificity"
            ],

        "unknown_auc":
            final_unknown[
                "auc"
            ],

        "prefix_variance":
            model.prefix_variance(),

        **stats,

        "use_class_weight":
            USE_CLASS_WEIGHT,

        "pos_weight":
            pos_weight_value,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "lr":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "n_trainable_parameters":
            n_trainable,
    }

    pd.DataFrame(
        [metrics]
    ).to_csv(
        out_dir
        /
        "final_metrics.csv",
        index=False,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(
        {
            "model_name":
                MODEL_NAME,

            "unknown_prefix_rate":
                unknown_prefix_rate,

            "unknown_group_fraction":
                UNKNOWN_GROUP_FRACTION,

            "known_validation_fraction":
                KNOWN_VALIDATION_FRACTION,

            "best_epoch":
                best_epoch,

            "n_known_groups":
                n_known_groups,

            "prefix_length":
                PREFIX_LENGTH,

            "prefix_dim":
                PREFIX_DIM,

            "prefix_insert_after_layer":
                PREFIX_INSERT_AFTER_LAYER,

            "group_prefixes":
                model.group_prefixes
                .detach()
                .cpu(),

            "raw_cov_diag":
                model.raw_cov_diag
                .detach()
                .cpu(),

            "prefix_projection_state_dict":
                model.prefix_projection
                .state_dict(),

            "classifier_state_dict":
                model.classifier
                .state_dict(),

            "known_group_list":
                known_group_list,
        },

        out_dir
        /
        "best_model.pt",
    )

    return metrics


# ============================================================
# 18. RUN BOTH EXPERIMENTS
# ============================================================

all_results = []

for unknown_prefix_rate in (
    UNKNOWN_PREFIX_RATES
):

    result = train_one_model(
        unknown_prefix_rate=(
            unknown_prefix_rate
        )
    )

    all_results.append(
        result
    )


# ============================================================
# 19. FINAL COMPARISON
# ============================================================

comparison = (
    pd.DataFrame(
        all_results
    )
    .sort_values(
        "unknown_prefix_rate"
    )
    .reset_index(drop=True)
)

comparison.to_csv(
    OUT_DIR_BASE
    /
    "unknown_groups_comparison.csv",
    index=False,
)


print("\n" + "=" * 80)
print("FINAL UNKNOWN-GROUP COMPARISON")
print("=" * 80)

columns_to_print = [
    "unknown_prefix_rate",
    "best_epoch",
    "n_known_groups",
    "n_unknown_groups",
    "n_train",
    "n_unknown_test",
    "known_validation_balanced_accuracy",
    "known_validation_auc",
    "unknown_accuracy",
    "unknown_balanced_accuracy",
    "unknown_sensitivity",
    "unknown_specificity",
    "unknown_auc",
]

print(
    comparison[
        columns_to_print
    ].to_string(
        index=False
    )
)


# ============================================================
# 20. DIFFERENCE BETWEEN 10% AND 0%
# ============================================================

if (
    0.00
    in comparison[
        "unknown_prefix_rate"
    ].values
    and
    0.10
    in comparison[
        "unknown_prefix_rate"
    ].values
):

    baseline_row = (
        comparison[
            comparison[
                "unknown_prefix_rate"
            ]
            ==
            0.00
        ]
        .iloc[0]
    )

    dropout_row = (
        comparison[
            comparison[
                "unknown_prefix_rate"
            ]
            ==
            0.10
        ]
        .iloc[0]
    )

    print("\n" + "=" * 80)
    print("EFFECT OF 10% UNKNOWN-PREFIX TRAINING")
    print("=" * 80)

    for metric in [
        "unknown_accuracy",
        "unknown_balanced_accuracy",
        "unknown_sensitivity",
        "unknown_specificity",
        "unknown_auc",
    ]:

        difference = (
            dropout_row[metric]
            -
            baseline_row[metric]
        )

        print(
            f"{metric}: "
            f"{difference:+.6f}"
        )


print(
    "\nSaved results in:",
    OUT_DIR_BASE,
)