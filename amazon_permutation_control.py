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
    "permutation_group_prefix"
)

OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

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
TEST_SIZE = 0.20

USE_CLASS_WEIGHT = True
LAMBDA_GRID = [0.001]
# One virtual prefix token per group.
PREFIX_LENGTH = 1

# False: keep the RoBERTa encoder frozen.
# Train only group prefixes, classifier, and prior variance.
TRAIN_LAST_LAYER = False

EPSILON = 1e-7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print("Device:", DEVICE)


# ============================================================
# 2. LOAD METADATA AND BINARY TARGET
# ============================================================

meta = pd.read_csv(META_PATH)
print("\nMetadata shape:", meta.shape)
print("Columns:", meta.columns.tolist())

if TARGET_COL not in meta.columns:
    raise ValueError(f"Missing column: {TARGET_COL}")
if GROUP_COL not in meta.columns:
    raise ValueError(f"Missing column: {GROUP_COL}")

if TEXT_COL is None or TEXT_COL not in meta.columns:
    raise ValueError(
        "Text column not found. Set TEXT_COL in the configuration. "
        f"Available columns: {meta.columns.tolist()}"
    )

print("Text column:", TEXT_COL)

# Original target: 0 = negative, 1 = neutral, 2 = positive
meta = meta[meta[TARGET_COL].isin([0, 2])].copy()
meta[TEXT_COL] = meta[TEXT_COL].fillna("").astype(str)
meta = meta[meta[TEXT_COL].str.strip().ne("")].reset_index(drop=True)

# Binary target: 0 = negative, 1 = positive
meta["y_binary"] = (meta[TARGET_COL] == 2).astype(np.float32)

# Encode groups as integers 0, 1, ..., G-1.
meta["group_id"] = pd.Categorical(meta[GROUP_COL]).codes.astype(np.int64)
n_groups = meta["group_id"].nunique()

texts = meta[TEXT_COL].tolist()
y = meta["y_binary"].to_numpy(dtype=np.float32)
group_ids = meta["group_id"].to_numpy(dtype=np.int64)

print("\nAfter filtering:", meta.shape)
print("Target distribution:")
print(meta["y_binary"].value_counts().sort_index())
print("Number of groups:", n_groups)
print("Reviews per group:")
print(meta.groupby("group_id").size().describe())


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(meta))
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

print("\nTrain size:", len(train_idx))
print("Test size:", len(test_idx))
print("Train groups:", len(np.unique(group_ids[train_idx])))
print("Test groups:", len(np.unique(group_ids[test_idx])))

# PERMUTATION CHECK: same labels, texts, split and group size distributions.
# Only the group ID received by the model is randomly reassigned.
PERM_SEED = 123
real_group_ids = group_ids.copy()
perm_rng = np.random.default_rng(PERM_SEED)
group_ids = group_ids.copy()
group_ids[train_idx] = perm_rng.permutation(real_group_ids[train_idx])
group_ids[test_idx] = perm_rng.permutation(real_group_ids[test_idx])
assert set(group_ids[test_idx]).issubset(set(group_ids[train_idx]))
assert np.array_equal(
    np.bincount(real_group_ids[train_idx], minlength=n_groups),
    np.bincount(group_ids[train_idx], minlength=n_groups))
assert np.array_equal(
    np.bincount(real_group_ids[test_idx], minlength=n_groups),
    np.bincount(group_ids[test_idx], minlength=n_groups))
print("Unchanged ID fraction train:",
      np.mean(group_ids[train_idx] == real_group_ids[train_idx]))
print("Unchanged ID fraction test:",
      np.mean(group_ids[test_idx] == real_group_ids[test_idx]))

pd.concat([
    pd.DataFrame({"global_idx": meta.iloc[train_idx]["global_idx"].to_numpy(),
                  "split": "train", "original_group_id": real_group_ids[train_idx],
                  "permuted_group_id": group_ids[train_idx]}),
    pd.DataFrame({"global_idx": meta.iloc[test_idx]["global_idx"].to_numpy(),
                  "split": "test", "original_group_id": real_group_ids[test_idx],
                  "permuted_group_id": group_ids[test_idx]})
]).to_csv(OUT_DIR_BASE / "permutation_assignments.csv", index=False)


# ============================================================
# 4. TOKENIZER AND DATALOADER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


class ReviewDataset(Dataset):
    def __init__(self, subset_indices):
        self.indices = np.asarray(subset_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = int(self.indices[position])
        return texts[idx], y[idx], group_ids[idx], idx


def collate_fn(batch):
    batch_texts = [row[0] for row in batch]
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
        torch.tensor([row[1] for row in batch], dtype=torch.float32),
        torch.tensor([row[2] for row in batch], dtype=torch.long),
        torch.tensor([row[3] for row in batch], dtype=torch.long),
    )


train_loader = DataLoader(
    ReviewDataset(train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    ReviewDataset(test_idx),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# ============================================================
# 5. MODEL
# ============================================================
#
# First 11 RoBERTa layers:
#     text -> contextual hidden states
#
# Before the final (12th) layer:
#     [group prefix; text hidden states]
#
# The prefix interacts with text through final-layer self-attention
# and is excluded from the final mean pooling.
# ============================================================


class LastLayerPrefixClassifier(nn.Module):
    def __init__(self, n_groups, prefix_length=1, train_last_layer=False):
        super().__init__()

        # Eager attention supports calling the Transformer layer directly.
        # Fallback for older Transformers versions.
        try:
            self.roberta = RobertaModel.from_pretrained(
                MODEL_NAME,
                add_pooling_layer=False,
                attn_implementation="eager",
            )
        except TypeError:
            self.roberta = RobertaModel.from_pretrained(
                MODEL_NAME,
                add_pooling_layer=False,
            )

        self.hidden_size = self.roberta.config.hidden_size
        self.n_groups = n_groups
        self.prefix_length = prefix_length
        self.train_last_layer = train_last_layer

        # Freeze RoBERTa.
        for parameter in self.roberta.parameters():
            parameter.requires_grad = False

        # Optionally train the last Transformer layer.
        if self.train_last_layer:
            for parameter in self.roberta.encoder.layer[-1].parameters():
                parameter.requires_grad = True

        # One trainable virtual token per group.
        self.group_prefixes = nn.Parameter(
            torch.zeros(n_groups, prefix_length, self.hidden_size)
        )
        nn.init.normal_(self.group_prefixes, mean=0.0, std=0.02)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Trainable scalar variance of the Gaussian prior.
        self.raw_cov_diag = nn.Parameter(torch.zeros(1))

    def train(self, mode=True):
        super().train(mode)

        # Keep the frozen backbone in evaluation mode.
        self.roberta.eval()

        if self.train_last_layer:
            self.roberta.encoder.layer[-1].train(mode)

        return self

    def centered_prefixes(self):
        return self.group_prefixes - self.group_prefixes.mean(dim=0, keepdim=True)

    def run_first_layers(self, input_ids, attention_mask):
        """
        Run RoBERTa and retrieve the output after layer 11,
        used as input to the final Transformer layer.
        """

        with torch.no_grad():
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden = outputs.hidden_states[-2]

        if hidden.ndim != 3:
            raise RuntimeError(
                "Invalid hidden-state shape. "
                f"Received {tuple(hidden.shape)}, "
                "expected [batch, sequence_length, hidden_size]."
            )

        return hidden

    @staticmethod
    def mean_pool(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    def run_last_layer(self, hidden, attention_mask, prefix=None):

        if hidden.ndim != 3:
            raise RuntimeError(
                "Invalid hidden-state shape for the final layer: "
                f"{tuple(hidden.shape)}"
            )

        if prefix is None:
            last_input = hidden
            last_mask = attention_mask
            text_start = 0
        else:
            prefix_mask = torch.ones(
                hidden.size(0),
                self.prefix_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            last_input = torch.cat([prefix, hidden], dim=1)
            last_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            text_start = self.prefix_length

        extended_last_mask, _ = self.roberta._create_attention_masks(
            attention_mask=last_mask,
            encoder_attention_mask=None,
            embedding_output=last_input,
            encoder_hidden_states=None,
            past_key_values=None,
        )

        last_hidden = self.roberta.encoder.layer[-1](
            last_input,
            attention_mask=extended_last_mask,
        )

        # Remove the prefix before pooling.
        text_hidden = last_hidden[:, text_start:, :]
        pooled = self.mean_pool(text_hidden, attention_mask)
        return self.classifier(pooled).squeeze(1)

    def forward(
        self,
        input_ids,
        attention_mask,
        group_ids_batch,
        return_without_prefix=False,
    ):
        hidden = self.run_first_layers(input_ids, attention_mask)
        prefix = self.centered_prefixes()[group_ids_batch]

        logits_with_prefix = self.run_last_layer(
            hidden,
            attention_mask,
            prefix=prefix,
        )

        if not return_without_prefix:
            return logits_with_prefix

        logits_without_prefix = self.run_last_layer(
            hidden,
            attention_mask,
            prefix=None,
        )

        return logits_with_prefix, logits_without_prefix

    def prefix_penalty(self):
        """Isotropic Gaussian prior: p_g ~ N(0, sigma^2 I)."""
        prefix = self.centered_prefixes()
        var = F.softplus(self.raw_cov_diag) + EPSILON

        mahalanobis = (prefix.pow(2) / var).mean()
        logdet = torch.log(var).sum()

        return mahalanobis + logdet

    def prefix_variance(self):
        var = F.softplus(self.raw_cov_diag) + EPSILON
        return var.detach().cpu().item()

    def prefix_stats(self):
        prefix = self.centered_prefixes().detach().reshape(self.n_groups, -1)
        norms = torch.linalg.vector_norm(prefix, dim=1)
        return {
            "prefix_norm_mean": norms.mean().cpu().item(),
            "prefix_norm_min": norms.min().cpu().item(),
            "prefix_norm_max": norms.max().cpu().item(),
            "prefix_value_std": prefix.std(unbiased=False).cpu().item(),
        }


# ============================================================
# 6. EVALUATION
# ============================================================


def calculate_metrics(y_true, logits):
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, pos_label=1),
        "specificity": recall_score(y_true, pred, pos_label=0),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]),
        "logits": logits,
        "prob": prob,
        "pred": pred,
        "y_true": y_true,
    }


@torch.no_grad()
def evaluate(model, loader, compare_without_prefix=False):
    model.eval()

    logits_with_all = []
    logits_without_all = []
    labels_all = []
    indices_all = []

    for input_ids, attention_mask, labels, groups, original_idx in loader:
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        groups = groups.to(DEVICE, non_blocking=True)

        if compare_without_prefix:
            logits_with, logits_without = model(
                input_ids,
                attention_mask,
                groups,
                return_without_prefix=True,
            )
            logits_without_all.append(logits_without.cpu())
        else:
            logits_with = model(input_ids, attention_mask, groups)

        logits_with_all.append(logits_with.cpu())
        labels_all.append(labels)
        indices_all.append(original_idx)

    logits_with = torch.cat(logits_with_all).numpy()
    labels = torch.cat(labels_all).numpy().astype(int)
    original_indices = torch.cat(indices_all).numpy().astype(int)

    result = calculate_metrics(labels, logits_with)
    result["original_idx"] = original_indices

    if compare_without_prefix:
        logits_without = torch.cat(logits_without_all).numpy()
        result_without = calculate_metrics(labels, logits_without)
        result["without_prefix"] = result_without
        result["logit_impact"] = logits_with - logits_without
        result["prob_impact"] = result["prob"] - result_without["prob"]

    return result


# ============================================================
# 7. SAVE / RESTORE TRAINABLE PARAMETERS
# ============================================================


def save_trainable_state(model):
    state = {
        "group_prefixes": model.group_prefixes.detach().cpu().clone(),
        "raw_cov_diag": model.raw_cov_diag.detach().cpu().clone(),
        "classifier": copy.deepcopy(model.classifier.state_dict()),
    }
    if TRAIN_LAST_LAYER:
        state["last_layer"] = copy.deepcopy(
            model.roberta.encoder.layer[-1].state_dict()
        )
    return state


def load_trainable_state(model, state):
    with torch.no_grad():
        model.group_prefixes.copy_(state["group_prefixes"].to(DEVICE))
        model.raw_cov_diag.copy_(state["raw_cov_diag"].to(DEVICE))
    model.classifier.load_state_dict(state["classifier"])
    if TRAIN_LAST_LAYER:
        model.roberta.encoder.layer[-1].load_state_dict(state["last_layer"])


# ============================================================
# 8. TRAIN FOR ONE REGULARIZATION STRENGTH
# ============================================================


def train_one_lambda(lambda_random):
    print("\n" + "=" * 80)
    print(f"LAMBDA_RANDOM = {lambda_random}")
    print("=" * 80)

    lambda_name = str(lambda_random).replace(".", "p")
    out_dir = OUT_DIR_BASE / f"lambda_{lambda_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    model = LastLayerPrefixClassifier(
        n_groups=n_groups,
        prefix_length=PREFIX_LENGTH,
        train_last_layer=TRAIN_LAST_LAYER,
    ).to(DEVICE)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_parameters)
    print("Trainable parameters:", f"{n_trainable:,}")

    if USE_CLASS_WEIGHT:
        n_pos = y[train_idx].sum()
        n_neg = len(train_idx) - n_pos
        pos_weight_value = float(n_neg / n_pos)
        pos_weight = torch.tensor([pos_weight_value], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("pos_weight:", pos_weight_value)
    else:
        pos_weight_value = np.nan
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_balanced_accuracy = -np.inf
    best_epoch = None
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_losses = []
        fixed_losses = []
        prefix_losses = []

        for input_ids, attention_mask, labels, groups, _ in train_loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            groups = groups.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, attention_mask, groups)
            fixed_loss = criterion(logits, labels)
            prefix_loss = model.prefix_penalty()
            loss = fixed_loss + lambda_random * prefix_loss

            loss.backward()
            optimizer.step()

            total_losses.append(loss.item())
            fixed_losses.append(fixed_loss.item())
            prefix_losses.append(prefix_loss.item())

        test_metrics = evaluate(model, test_loader)
        stats = model.prefix_stats()

        row = {
            "lambda_random": lambda_random,
            "epoch": epoch,
            "train_loss": np.mean(total_losses),
            "train_fixed_loss": np.mean(fixed_losses),
            "train_prefix_loss": np.mean(prefix_losses),
            "prefix_variance": model.prefix_variance(),
            **stats,
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_auc": test_metrics["auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={row['train_loss']:.4f} | "
            f"fixed={row['train_fixed_loss']:.4f} | "
            f"prefix={row['train_prefix_loss']:.4f} | "
            f"var={row['prefix_variance']:.6f} | "
            f"norm={row['prefix_norm_mean']:.4f} | "
            f"acc={row['test_accuracy']:.4f} | "
            f"bal_acc={row['test_balanced_accuracy']:.4f} | "
            f"sens={row['test_sensitivity']:.4f} | "
            f"spec={row['test_specificity']:.4f} | "
            f"auc={row['test_auc']:.4f}"
        )

        if test_metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = test_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = save_trainable_state(model)

    load_trainable_state(model, best_state)
    final = evaluate(model, test_loader, compare_without_prefix=True)
    without = final["without_prefix"]

    print("\nBEST MODEL")
    print("Best epoch:", best_epoch)
    print("Balanced accuracy with prefix:", final["balanced_accuracy"])
    print("Balanced accuracy without prefix:", without["balanced_accuracy"])
    print("Confusion matrix with prefix:")
    print(final["confusion_matrix"])

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------
    pd.DataFrame(history).to_csv(
        out_dir / "training_history.csv", index=False
    )

    # --------------------------------------------------------
    # Test predictions and prefix impact
    # --------------------------------------------------------
    pred_df = meta.iloc[final["original_idx"]].copy().reset_index(drop=True)
    pred_df["prob_with_prefix"] = final["prob"]
    pred_df["pred_with_prefix"] = final["pred"]
    pred_df["prob_without_prefix"] = without["prob"]
    pred_df["pred_without_prefix"] = without["pred"]
    pred_df["prefix_logit_impact"] = final["logit_impact"]
    pred_df["prefix_prob_impact"] = final["prob_impact"]
    pred_df["y_true"] = final["y_true"]
    pred_df["permuted_group_id"] = group_ids[final["original_idx"]]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Group summaries by original named products are deliberately omitted:
    # permuted group IDs are not associated with real products.

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------
    stats = model.prefix_stats()
    metrics = {
        "model": "group_prefix_permuted_product_ids",
        "group_col": GROUP_COL,
        "lambda_random": lambda_random,
        "best_epoch": best_epoch,
        "n_groups": n_groups,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "prefix_length": PREFIX_LENGTH,
        "train_last_layer": TRAIN_LAST_LAYER,
        "accuracy": final["accuracy"],
        "balanced_accuracy": final["balanced_accuracy"],
        "sensitivity": final["sensitivity"],
        "specificity": final["specificity"],
        "auc": final["auc"],
        "balanced_accuracy_without_prefix": without["balanced_accuracy"],
        "auc_without_prefix": without["auc"],
        "mean_abs_logit_impact": np.mean(np.abs(final["logit_impact"])),
        "mean_abs_prob_impact": np.mean(np.abs(final["prob_impact"])),
        "prefix_variance": model.prefix_variance(),
        **stats,
        "use_class_weight": USE_CLASS_WEIGHT,
        "pos_weight": pos_weight_value,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "n_trainable_parameters": n_trainable,
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "final_metrics.csv", index=False)

    # Save trainable parameters separately from the pretrained RoBERTa model.
    torch.save(
        {
            "model_name": MODEL_NAME,
            "n_groups": n_groups,
            "prefix_length": PREFIX_LENGTH,
            "group_prefixes": model.group_prefixes.detach().cpu(),
            "raw_cov_diag": model.raw_cov_diag.detach().cpu(),
            "classifier_state_dict": model.classifier.state_dict(),
            "last_layer_state_dict": (
                model.roberta.encoder.layer[-1].state_dict()
                if TRAIN_LAST_LAYER
                else None
            ),
        },
        out_dir / "best_model.pt",
    )

    return metrics


# ============================================================
# 9. GRID SEARCH
# ============================================================


grid_results = []

for lambda_random in LAMBDA_GRID:
    grid_results.append(train_one_lambda(lambda_random))

grid_df = pd.DataFrame(grid_results).sort_values(
    "balanced_accuracy", ascending=False
)
grid_df.to_csv(OUT_DIR_BASE / "grid_summary.csv", index=False)

print("\n" + "=" * 80)
print("GRID SEARCH SUMMARY")
print("=" * 80)
print(
    grid_df[
        [
            "lambda_random",
            "best_epoch",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "auc",
            "balanced_accuracy_without_prefix",
            "mean_abs_logit_impact",
            "prefix_variance",
            "prefix_norm_mean",
        ]
    ]
)

print("\nBest lambda:")
print(grid_df.iloc[0])
print("\nSaved:", OUT_DIR_BASE / "grid_summary.csv")