import argparse
import gc
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, PreTrainedTokenizerFast


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path("/work/u10767335/data/breath")

TRAIN_PATH = BASE / "breath_train_4class_preprocessed512_nodiag.csv.gz"
TEST_PATH = BASE / "breath_test_4class_preprocessed512_nodiag.csv.gz"

OUT_BASE = BASE / "breath_unseen_localities_loo"
OUT_BASE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
MODEL_REVISION = "e934c95baeee3f4e613d426d044e52baa82300df"

TEXT_COL = "testo_512"
TARGET_COL = "diagnosi"
GROUP_COL = "localita"
PATIENT_COL = "pedianet_id"
DOC_ID_COL = "guiddocumento"

CLASS_NAMES = [
    "bronchiolite",
    "bronchite",
    "laringite",
    "wheezing",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
BATCH_SIZE = 8
MAX_LENGTH = 512
NUM_WORKERS = 2
USE_CLASS_WEIGHT = True

# Hyperparameters are FIXED from the main BREATH experiment.
# No tuning is performed on held-out localities.
FIXED_EPOCHS = 5
PREFIX_EPOCHS = 5
LAMBDA_PREFIX = 0.01
ZERO_PREFIX_MASK_RATE = 0.10

LR_CLASSIFIER = 1e-3
LR_PREFIX = 1e-3
LR_LAST_LAYER = 2e-5
WEIGHT_DECAY = 1e-4

PREFIX_LENGTH = 1
PREFIX_DIM = 64
PREFIX_INSERT_AFTER_LAYER = 11
EPSILON = 1e-7

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 123

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_data():
    train = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
    test = pd.read_csv(TEST_PATH).reset_index(drop=True)

    required = {
        TEXT_COL,
        TARGET_COL,
        GROUP_COL,
        PATIENT_COL,
        DOC_ID_COL,
    }

    for name, df in [("train", train), ("test", test)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")

        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
        if df[TEXT_COL].str.strip().eq("").any():
            raise ValueError(f"{name}: empty texts found in {TEXT_COL}")

        unknown = set(df[TARGET_COL].astype(str)) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"{name}: unexpected labels {sorted(unknown)}")

        if df[GROUP_COL].isna().any():
            raise ValueError(f"{name}: missing localities")
        if df[PATIENT_COL].isna().any():
            raise ValueError(f"{name}: missing patient IDs")

    train["y_class"] = (
        train[TARGET_COL].astype(str).map(CLASS_TO_ID).astype(np.int64)
    )
    test["y_class"] = (
        test[TARGET_COL].astype(str).map(CLASS_TO_ID).astype(np.int64)
    )

    train["original_train_row_id"] = np.arange(len(train), dtype=np.int64)
    test["original_test_row_id"] = np.arange(len(test), dtype=np.int64)

    overlap = set(train[PATIENT_COL]) & set(test[PATIENT_COL])
    if overlap:
        raise ValueError(
            f"Original saved train/test split is not patient-disjoint: {len(overlap)} overlap"
        )

    return train, test


# ============================================================
# 4. TOKENIZER
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
# 5. DATASETS / LOADERS
# ============================================================

class FixedDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            str(row[TEXT_COL]),
            int(row["y_class"]),
            int(row["original_test_row_id"]) if "original_test_row_id" in row else -1,
        )


class PrefixDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            str(row[TEXT_COL]),
            int(row["y_class"]),
            int(row["group_id"]) if "group_id" in row else -1,
            int(row["original_test_row_id"]) if "original_test_row_id" in row else -1,
        )


def tokenize_texts(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def collate_fixed(batch):
    encoded = tokenize_texts([x[0] for x in batch])
    return (
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor([x[1] for x in batch], dtype=torch.long),
        torch.tensor([x[2] for x in batch], dtype=torch.long),
    )


def collate_prefix(batch):
    encoded = tokenize_texts([x[0] for x in batch])
    return (
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor([x[1] for x in batch], dtype=torch.long),
        torch.tensor([x[2] for x in batch], dtype=torch.long),
        torch.tensor([x[3] for x in batch], dtype=torch.long),
    )


def make_fixed_loader(df, shuffle, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        FixedDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fixed,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=gen if shuffle else None,
    )


def make_prefix_loader(df, shuffle, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        PrefixDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_prefix,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=gen if shuffle else None,
    )


# ============================================================
# 6. CLASS WEIGHTS
# ============================================================

def make_criterion(labels):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    if np.any(counts == 0):
        raise ValueError(f"Training fold is missing a class: {counts.tolist()}")

    if not USE_CLASS_WEIGHT:
        return nn.CrossEntropyLoss(), counts, None

    weights = len(labels) / (NUM_CLASSES * counts.astype(np.float64))
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    return nn.CrossEntropyLoss(weight=weights_t), counts, weights


# ============================================================
# 7. FIXED-ONLY MODEL
# ============================================================

class FixedOnlyClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            use_safetensors=True,
        )
        self.hidden_size = self.encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad = False

        for p in self.encoder.encoder.layer[-1].parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.encoder.encoder.layer[-1].train(mode)
        return self

    @staticmethod
    def mean_pool(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)


def fixed_optimizer(model):
    return torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": LR_CLASSIFIER},
            {
                "params": model.encoder.encoder.layer[-1].parameters(),
                "lr": LR_LAST_LAYER,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# 8. GROUP-PREFIX MODEL
# ============================================================

class UmbertoGroupPrefixClassifier(nn.Module):
    def __init__(self, n_groups):
        super().__init__()

        self.umberto = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            use_safetensors=True,
            attn_implementation="eager",
        )

        self.hidden_size = self.umberto.config.hidden_size
        self.num_layers = len(self.umberto.encoder.layer)

        if self.num_layers != 12:
            raise ValueError(f"Expected 12 layers, found {self.num_layers}")
        if PREFIX_INSERT_AFTER_LAYER != self.num_layers - 1:
            raise ValueError("Prefix must be inserted immediately before final layer")

        self.n_groups = n_groups

        for p in self.umberto.parameters():
            p.requires_grad = False
        for p in self.umberto.encoder.layer[-1].parameters():
            p.requires_grad = True

        self.group_prefixes = nn.Parameter(
            torch.zeros(n_groups, PREFIX_LENGTH, PREFIX_DIM)
        )
        nn.init.normal_(self.group_prefixes, mean=0.0, std=0.02)

        self.prefix_projection = nn.Linear(PREFIX_DIM, self.hidden_size, bias=False)
        nn.init.normal_(self.prefix_projection.weight, mean=0.0, std=0.02)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )

        self.raw_cov_diag = nn.Parameter(torch.zeros(1))

    def train(self, mode=True):
        super().train(mode)
        self.umberto.eval()
        self.umberto.encoder.layer[-1].train(mode)
        return self

    def centered_prefixes(self):
        return self.group_prefixes - self.group_prefixes.mean(dim=0, keepdim=True)

    def projected_prefixes(self):
        return self.prefix_projection(self.centered_prefixes())

    def _extended_attention_mask(self, attention_mask, hidden):
        try:
            result = self.umberto._create_attention_masks(
                attention_mask=attention_mask,
                encoder_attention_mask=None,
                embedding_output=hidden,
                encoder_hidden_states=None,
                past_key_values=None,
            )
            return result[0] if isinstance(result, tuple) else result
        except (AttributeError, TypeError):
            extended = attention_mask[:, None, None, :].to(dtype=hidden.dtype)
            return (1.0 - extended) * torch.finfo(hidden.dtype).min

    @staticmethod
    def _hidden(layer_output):
        if isinstance(layer_output, tuple):
            return layer_output[0]
        if hasattr(layer_output, "last_hidden_state"):
            return layer_output.last_hidden_state
        return layer_output

    def encode_first_11(self, input_ids, attention_mask):
        with torch.no_grad():
            hidden = self.umberto.embeddings(input_ids=input_ids)
            for layer_idx in range(PREFIX_INSERT_AFTER_LAYER):
                extended_mask = self._extended_attention_mask(attention_mask, hidden)
                out = self.umberto.encoder.layer[layer_idx](
                    hidden,
                    attention_mask=extended_mask,
                )
                hidden = self._hidden(out)
        return hidden

    def run_last_layer(self, hidden, attention_mask):
        extended_mask = self._extended_attention_mask(attention_mask, hidden)
        out = self.umberto.encoder.layer[-1](
            hidden,
            attention_mask=extended_mask,
        )
        return self._hidden(out)

    @staticmethod
    def mean_pool(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def get_prefix(self, group_ids, force_zero_mask=None, all_zero=False):
        batch_size = group_ids.size(0)

        if all_zero:
            # Important: a zero-valued prefix TOKEN remains in the sequence.
            return torch.zeros(
                batch_size,
                PREFIX_LENGTH,
                self.hidden_size,
                dtype=self.group_prefixes.dtype,
                device=group_ids.device,
            )

        if torch.any(group_ids < 0):
            raise ValueError("Unknown group IDs require all_zero=True")

        prefix = self.projected_prefixes()[group_ids]

        if force_zero_mask is not None:
            mask = force_zero_mask.bool().view(-1, 1, 1)
            prefix = torch.where(mask, torch.zeros_like(prefix), prefix)

        return prefix

    def forward(
        self,
        input_ids,
        attention_mask,
        group_ids,
        force_zero_mask=None,
        all_zero=False,
    ):
        hidden = self.encode_first_11(input_ids, attention_mask)

        prefix = self.get_prefix(
            group_ids,
            force_zero_mask=force_zero_mask,
            all_zero=all_zero,
        )

        prefix_mask = torch.ones(
            hidden.size(0),
            PREFIX_LENGTH,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        combined_hidden = torch.cat([prefix, hidden], dim=1)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        last_hidden = self.run_last_layer(combined_hidden, combined_mask)
        text_hidden = last_hidden[:, PREFIX_LENGTH:, :]
        pooled = self.mean_pool(text_hidden, attention_mask)
        return self.classifier(pooled)

    def prefix_penalty(self):
        prefix = self.centered_prefixes()
        var = F.softplus(self.raw_cov_diag) + EPSILON
        mahalanobis = (prefix.pow(2) / var).mean()
        logdet = torch.log(var).sum()
        return mahalanobis + logdet


def prefix_optimizer(model):
    prefix_parameters = [model.group_prefixes, model.raw_cov_diag]
    prefix_parameters += list(model.prefix_projection.parameters())

    return torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": LR_CLASSIFIER},
            {"params": prefix_parameters, "lr": LR_PREFIX},
            {
                "params": model.umberto.encoder.layer[-1].parameters(),
                "lr": LR_LAST_LAYER,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# 9. METRICS
# ============================================================

def probabilities_from_logits(logits):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits_t, dim=1).numpy()


def multiclass_brier(y_true, probs):
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(NUM_CLASSES, dtype=float)[y_true]
    return float(np.mean((probs - onehot) ** 2))


def metrics_from_probs(y_true, probs):
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)

    recalls = recall_score(
        y_true,
        pred,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )

    try:
        if len(np.unique(y_true)) == NUM_CLASSES:
            macro_auc = roc_auc_score(
                y_true,
                probs,
                labels=np.arange(NUM_CLASSES),
                multi_class="ovr",
                average="macro",
            )
            weighted_auc = roc_auc_score(
                y_true,
                probs,
                labels=np.arange(NUM_CLASSES),
                multi_class="ovr",
                average="weighted",
            )
        else:
            macro_auc = np.nan
            weighted_auc = np.nan
    except ValueError:
        macro_auc = np.nan
        weighted_auc = np.nan

    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(
            y_true,
            pred,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            pred,
            labels=np.arange(NUM_CLASSES),
            average="weighted",
            zero_division=0,
        ),
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "brier": multiclass_brier(y_true, probs),
        "pred": pred,
    }

    for i, name in ID_TO_CLASS.items():
        out[f"recall_{name}"] = float(recalls[i])

    return out


# ============================================================
# 10. TRAINING / EVALUATION
# ============================================================

def train_fixed_model(train_df, seed):
    set_seed(seed)
    loader = make_fixed_loader(train_df, shuffle=True, seed=seed)
    criterion, counts, weights = make_criterion(train_df["y_class"].to_numpy())

    model = FixedOnlyClassifier().to(DEVICE)
    optimizer = fixed_optimizer(model)

    history = []

    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        losses = []

        for input_ids, attention_mask, labels, _ in loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"    Fixed epoch {epoch:02d}/{FIXED_EPOCHS}: loss={mean_loss:.4f}")

    return model, pd.DataFrame(history), counts, weights


def train_prefix_model(train_df, n_groups, seed, zero_mask_rate):
    set_seed(seed)
    loader = make_prefix_loader(train_df, shuffle=True, seed=seed)
    criterion, counts, weights = make_criterion(train_df["y_class"].to_numpy())

    model = UmbertoGroupPrefixClassifier(n_groups=n_groups).to(DEVICE)
    optimizer = prefix_optimizer(model)

    history = []

    for epoch in range(1, PREFIX_EPOCHS + 1):
        model.train()
        total_losses = []
        ce_losses = []
        penalty_losses = []
        n_zero = 0
        n_seen = 0

        for input_ids, attention_mask, labels, groups, _ in loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            groups = groups.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if zero_mask_rate > 0:
                zero_mask = torch.rand(groups.size(0), device=DEVICE) < zero_mask_rate
            else:
                zero_mask = torch.zeros(groups.size(0), dtype=torch.bool, device=DEVICE)

            logits = model(
                input_ids,
                attention_mask,
                groups,
                force_zero_mask=zero_mask,
                all_zero=False,
            )

            ce = criterion(logits, labels)
            penalty = model.prefix_penalty()
            loss = ce + LAMBDA_PREFIX * penalty

            loss.backward()
            optimizer.step()

            total_losses.append(loss.item())
            ce_losses.append(ce.item())
            penalty_losses.append(penalty.item())
            n_zero += int(zero_mask.sum().item())
            n_seen += int(groups.size(0))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(total_losses)),
            "train_ce_loss": float(np.mean(ce_losses)),
            "train_prefix_loss": float(np.mean(penalty_losses)),
            "actual_zero_prefix_rate": n_zero / max(n_seen, 1),
        }
        history.append(row)
        print(
            f"    Prefix mask={zero_mask_rate:.2f} epoch {epoch:02d}/{PREFIX_EPOCHS}: "
            f"loss={row['train_loss']:.4f} ce={row['train_ce_loss']:.4f} "
            f"prefix={row['train_prefix_loss']:.4f} "
            f"zero_rate={row['actual_zero_prefix_rate']:.3f}"
        )

    return model, pd.DataFrame(history), counts, weights


@torch.no_grad()
def evaluate_fixed(model, test_df, seed):
    loader = make_fixed_loader(test_df, shuffle=False, seed=seed)
    model.eval()

    logits_all = []
    y_all = []
    row_all = []

    for input_ids, attention_mask, labels, rows in loader:
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        logits = model(input_ids, attention_mask)

        logits_all.append(logits.cpu())
        y_all.append(labels)
        row_all.append(rows)

    logits = torch.cat(logits_all).numpy()
    y = torch.cat(y_all).numpy().astype(np.int64)
    rows = torch.cat(row_all).numpy().astype(np.int64)
    probs = probabilities_from_logits(logits)
    return y, probs, rows


@torch.no_grad()
def evaluate_prefix_zero(model, test_df, seed):
    # test_df has group_id=-1 because its locality is unseen.
    loader = make_prefix_loader(test_df, shuffle=False, seed=seed)
    model.eval()

    logits_all = []
    y_all = []
    row_all = []

    for input_ids, attention_mask, labels, groups, rows in loader:
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        groups = groups.to(DEVICE, non_blocking=True)

        logits = model(
            input_ids,
            attention_mask,
            groups,
            all_zero=True,
        )

        logits_all.append(logits.cpu())
        y_all.append(labels)
        row_all.append(rows)

    logits = torch.cat(logits_all).numpy()
    y = torch.cat(y_all).numpy().astype(np.int64)
    rows = torch.cat(row_all).numpy().astype(np.int64)
    probs = probabilities_from_logits(logits)
    return y, probs, rows


# ============================================================
# 11. ONE LEAVE-ONE-LOCALITY-OUT FOLD
# ============================================================

def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return text or "locality"


def run_fold(fold_index):
    train_full, test_full = load_data()

    localities = sorted(test_full[GROUP_COL].astype(str).unique().tolist())
    train_localities = set(train_full[GROUP_COL].astype(str).unique())

    missing_train = set(localities) - train_localities
    if missing_train:
        raise ValueError(f"Test localities absent from original training set: {missing_train}")

    if fold_index < 0 or fold_index >= len(localities):
        raise ValueError(f"fold must be 0..{len(localities)-1}")

    heldout = localities[fold_index]
    fold_seed = SEED + fold_index

    train_df = train_full[train_full[GROUP_COL].astype(str) != heldout].copy().reset_index(drop=True)
    test_df = test_full[test_full[GROUP_COL].astype(str) == heldout].copy().reset_index(drop=True)

    if len(test_df) == 0:
        raise ValueError(f"No test documents for held-out locality {heldout}")

    # Original external test is already patient-disjoint from original training.
    overlap = set(train_df[PATIENT_COL]) & set(test_df[PATIENT_COL])
    if overlap:
        raise ValueError(f"Patient overlap in fold {heldout}: {len(overlap)}")

    group_names = sorted(train_df[GROUP_COL].astype(str).unique().tolist())
    group_to_id = {g: i for i, g in enumerate(group_names)}

    train_df["group_id"] = train_df[GROUP_COL].astype(str).map(group_to_id).astype(np.int64)
    test_df["group_id"] = -1

    fold_dir = OUT_BASE / f"fold_{fold_index:02d}_{safe_name(heldout)}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"FOLD {fold_index:02d} / {len(localities)-1:02d} — HELD-OUT LOCALITY: {heldout}")
    print("=" * 100)
    print("Device:", DEVICE)
    print("Training documents:", len(train_df))
    print("Training patients:", train_df[PATIENT_COL].nunique())
    print("Training localities:", train_df[GROUP_COL].nunique())
    print("Held-out test documents:", len(test_df))
    print("Held-out test patients:", test_df[PATIENT_COL].nunique())
    print("Test class counts:")
    print(test_df[TARGET_COL].value_counts().reindex(CLASS_NAMES, fill_value=0))

    # ---------------- FIXED ONLY ----------------
    print("\n[1/3] FIXED-ONLY")
    fixed_model, fixed_hist, fixed_counts, fixed_weights = train_fixed_model(train_df, fold_seed)
    y, p_fixed, rows = evaluate_fixed(fixed_model, test_df, fold_seed)
    m_fixed = metrics_from_probs(y, p_fixed)
    fixed_hist.to_csv(fold_dir / "fixed_training_history.csv", index=False)

    del fixed_model
    cleanup_cuda()

    # ---------------- PREFIX STANDARD ----------------
    print("\n[2/3] GROUP-PREFIX — STANDARD TRAINING")
    prefix_std, std_hist, _, _ = train_prefix_model(
        train_df,
        n_groups=len(group_names),
        seed=fold_seed,
        zero_mask_rate=0.0,
    )
    y_std, p_std, rows_std = evaluate_prefix_zero(prefix_std, test_df, fold_seed)
    if not np.array_equal(y, y_std) or not np.array_equal(rows, rows_std):
        raise RuntimeError("Fixed and standard-prefix test outputs are not aligned")
    m_std = metrics_from_probs(y, p_std)
    std_hist.to_csv(fold_dir / "prefix_standard_training_history.csv", index=False)

    del prefix_std
    cleanup_cuda()

    # ---------------- PREFIX MASKED ----------------
    print("\n[3/3] GROUP-PREFIX — 10% ZERO-PREFIX MASKING")
    prefix_masked, masked_hist, _, _ = train_prefix_model(
        train_df,
        n_groups=len(group_names),
        seed=fold_seed,
        zero_mask_rate=ZERO_PREFIX_MASK_RATE,
    )
    y_mask, p_mask, rows_mask = evaluate_prefix_zero(prefix_masked, test_df, fold_seed)
    if not np.array_equal(y, y_mask) or not np.array_equal(rows, rows_mask):
        raise RuntimeError("Fixed and masked-prefix test outputs are not aligned")
    m_mask = metrics_from_probs(y, p_mask)
    masked_hist.to_csv(fold_dir / "prefix_masked_training_history.csv", index=False)

    del prefix_masked
    cleanup_cuda()

    # ---------------- SAVE PREDICTIONS ----------------
    order = np.argsort(rows)
    rows = rows[order]
    y = y[order]
    p_fixed = p_fixed[order]
    p_std = p_std[order]
    p_mask = p_mask[order]

    pred_fixed = p_fixed.argmax(axis=1)
    pred_std = p_std.argmax(axis=1)
    pred_mask = p_mask.argmax(axis=1)

    original_lookup = test_full.set_index("original_test_row_id")
    out = original_lookup.loc[rows].reset_index()[
        [
            "original_test_row_id",
            DOC_ID_COL,
            PATIENT_COL,
            GROUP_COL,
            TARGET_COL,
            "y_class",
            "distance_days",
        ]
    ].copy()

    out["fold_index"] = fold_index
    out["heldout_locality"] = heldout

    out["pred_fixed_id"] = pred_fixed
    out["pred_fixed"] = [ID_TO_CLASS[int(x)] for x in pred_fixed]
    out["pred_prefix_standard_zero_id"] = pred_std
    out["pred_prefix_standard_zero"] = [ID_TO_CLASS[int(x)] for x in pred_std]
    out["pred_prefix_masked_zero_id"] = pred_mask
    out["pred_prefix_masked_zero"] = [ID_TO_CLASS[int(x)] for x in pred_mask]

    for c, name in enumerate(CLASS_NAMES):
        out[f"prob_fixed_{name}"] = p_fixed[:, c]
        out[f"prob_prefix_standard_zero_{name}"] = p_std[:, c]
        out[f"prob_prefix_masked_zero_{name}"] = p_mask[:, c]

    out.to_csv(fold_dir / "unseen_predictions.csv", index=False)

    # ---------------- SAVE METRICS ----------------
    metric_rows = []
    for model_name, metrics in [
        ("Fixed-only", m_fixed),
        ("Group-prefix standard (zero at test)", m_std),
        ("Group-prefix masked 10% (zero at test)", m_mask),
    ]:
        row = {
            "fold_index": fold_index,
            "heldout_locality": heldout,
            "model": model_name,
            "n_test_documents": len(test_df),
            "n_test_patients": test_df[PATIENT_COL].nunique(),
            "n_train_documents": len(train_df),
            "n_train_localities": len(group_names),
        }
        for key, value in metrics.items():
            if key != "pred":
                row[key] = value
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(fold_dir / "fold_metrics.csv", index=False)

    config = {
        "fold_index": fold_index,
        "heldout_locality": heldout,
        "seed": fold_seed,
        "fixed_epochs": FIXED_EPOCHS,
        "prefix_epochs": PREFIX_EPOCHS,
        "lambda_prefix": LAMBDA_PREFIX,
        "zero_prefix_mask_rate": ZERO_PREFIX_MASK_RATE,
        "prefix_dim": PREFIX_DIM,
        "prefix_length": PREFIX_LENGTH,
        "prefix_insert_after_layer": PREFIX_INSERT_AFTER_LAYER,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "class_names": CLASS_NAMES,
        "fixed_class_counts": fixed_counts.tolist(),
        "fixed_class_weights": None if fixed_weights is None else fixed_weights.tolist(),
    }
    with open(fold_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nFOLD RESULTS")
    print(metrics_df[[
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auc",
        "brier",
    ]].to_string(index=False))
    print("\nSaved in:", fold_dir)


# ============================================================
# 12. AGGREGATE ALL FOLDS
# ============================================================

def get_probs(df, prefix):
    cols = [f"{prefix}_{name}" for name in CLASS_NAMES]
    return df[cols].to_numpy(dtype=float)


def aggregate_model_metrics(df, pred_col, prob_prefix):
    y = df["y_class"].to_numpy(dtype=np.int64)
    probs = get_probs(df, prob_prefix)
    return metrics_from_probs(y, probs)


def patient_cluster_bootstrap_model_metrics(
    df,
    model_name,
    prob_prefix,
    n_boot=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
):
    """
    Patient-level cluster bootstrap 95% CIs for one model's aggregate metrics.
    Patients are sampled with replacement and all documents belonging to each
    sampled patient are retained.
    """
    rng = np.random.default_rng(seed)

    patients = df[PATIENT_COL].astype(str).unique()
    patient_rows = {
        patient: np.where(df[PATIENT_COL].astype(str).to_numpy() == patient)[0]
        for patient in patients
    }

    y_all = df["y_class"].to_numpy(dtype=np.int64)
    probs_all = get_probs(df, prob_prefix)
    point = metrics_from_probs(y_all, probs_all)

    tracked = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auc",
        "weighted_auc",
        "brier",
    ]
    samples = {metric: [] for metric in tracked}

    attempts = 0
    max_attempts = n_boot * 20

    while len(samples["accuracy"]) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled_patients = rng.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        idx = np.concatenate([patient_rows[p] for p in sampled_patients])

        y = y_all[idx]
        if len(np.unique(y)) != NUM_CLASSES:
            continue

        m = metrics_from_probs(y, probs_all[idx])
        if np.isnan(m["macro_auc"]) or np.isnan(m["weighted_auc"]):
            continue

        for metric in tracked:
            samples[metric].append(m[metric])

    if len(samples["accuracy"]) != n_boot:
        raise RuntimeError(
            f"Only obtained {len(samples['accuracy'])}/{n_boot} valid bootstrap replicates "
            f"for {model_name}"
        )

    rows = []
    for metric in tracked:
        arr = np.asarray(samples[metric], dtype=float)
        low, high = np.percentile(arr, [2.5, 97.5])
        rows.append({
            "model": model_name,
            "metric": metric,
            "estimate": point[metric],
            "ci95_low": low,
            "ci95_high": high,
            "n_bootstrap": n_boot,
            "n_documents": len(df),
            "n_patients": len(patients),
        })

    return pd.DataFrame(rows)


def patient_cluster_bootstrap_differences(df, model_a, model_b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Paired bootstrap by patient. Delta = model_b - model_a.
    For Brier, negative delta favours model_b.
    """
    rng = np.random.default_rng(seed)

    model_info = {
        "fixed": ("pred_fixed_id", "prob_fixed"),
        "standard": ("pred_prefix_standard_zero_id", "prob_prefix_standard_zero"),
        "masked": ("pred_prefix_masked_zero_id", "prob_prefix_masked_zero"),
    }

    pred_a_col, prob_a_prefix = model_info[model_a]
    pred_b_col, prob_b_prefix = model_info[model_b]

    patients = df[PATIENT_COL].astype(str).unique()
    patient_rows = {
        p: np.where(df[PATIENT_COL].astype(str).to_numpy() == p)[0]
        for p in patients
    }

    y_all = df["y_class"].to_numpy(dtype=np.int64)
    p_a_all = get_probs(df, prob_a_prefix)
    p_b_all = get_probs(df, prob_b_prefix)

    point_a = metrics_from_probs(y_all, p_a_all)
    point_b = metrics_from_probs(y_all, p_b_all)

    tracked = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "brier"]
    samples = {m: [] for m in tracked}

    attempts = 0
    max_attempts = n_boot * 20

    while len(samples["accuracy"]) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([patient_rows[p] for p in sampled_patients])

        y = y_all[idx]
        if len(np.unique(y)) != NUM_CLASSES:
            continue

        ma = metrics_from_probs(y, p_a_all[idx])
        mb = metrics_from_probs(y, p_b_all[idx])

        if np.isnan(ma["macro_auc"]) or np.isnan(mb["macro_auc"]):
            continue

        for metric in tracked:
            samples[metric].append(mb[metric] - ma[metric])

    if len(samples["accuracy"]) != n_boot:
        raise RuntimeError(
            f"Only obtained {len(samples['accuracy'])}/{n_boot} valid bootstrap replicates"
        )

    rows = []
    for metric in tracked:
        arr = np.asarray(samples[metric], dtype=float)
        low, high = np.percentile(arr, [2.5, 97.5])
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "metric_a": point_a[metric],
            "metric_b": point_b[metric],
            "delta_b_minus_a": point_b[metric] - point_a[metric],
            "ci95_low": low,
            "ci95_high": high,
            "n_bootstrap": n_boot,
        })

    return pd.DataFrame(rows)


def aggregate_results():
    train_full, test_full = load_data()
    localities = sorted(test_full[GROUP_COL].astype(str).unique().tolist())

    pred_parts = []
    metric_parts = []
    missing = []

    for fold_index, locality in enumerate(localities):
        fold_dir = OUT_BASE / f"fold_{fold_index:02d}_{safe_name(locality)}"
        pred_path = fold_dir / "unseen_predictions.csv"
        metrics_path = fold_dir / "fold_metrics.csv"

        if not pred_path.exists() or not metrics_path.exists():
            missing.append((fold_index, locality))
            continue

        pred_parts.append(pd.read_csv(pred_path))
        metric_parts.append(pd.read_csv(metrics_path))

    if missing:
        msg = "\n".join([f"  fold {i:02d}: {loc}" for i, loc in missing])
        raise RuntimeError(f"Missing fold outputs:\n{msg}")

    preds = pd.concat(pred_parts, ignore_index=True)
    fold_metrics = pd.concat(metric_parts, ignore_index=True)

    if preds["original_test_row_id"].duplicated().any():
        raise RuntimeError("Duplicate original test rows in aggregated predictions")

    if len(preds) != len(test_full):
        raise RuntimeError(
            f"Aggregated predictions contain {len(preds)} rows; expected {len(test_full)}"
        )

    preds = preds.sort_values("original_test_row_id").reset_index(drop=True)

    expected = np.arange(len(test_full), dtype=np.int64)
    if not np.array_equal(preds["original_test_row_id"].to_numpy(dtype=np.int64), expected):
        raise RuntimeError("Aggregated predictions do not cover the full external test set")

    aggregate_dir = OUT_BASE / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    preds.to_csv(aggregate_dir / "all_unseen_predictions.csv", index=False)
    fold_metrics.to_csv(aggregate_dir / "all_fold_metrics.csv", index=False)

    # ---------- GLOBAL METRICS ----------
    model_specs = [
        ("Fixed-only", "pred_fixed_id", "prob_fixed"),
        ("Group-prefix standard (zero at test)", "pred_prefix_standard_zero_id", "prob_prefix_standard_zero"),
        ("Group-prefix masked 10% (zero at test)", "pred_prefix_masked_zero_id", "prob_prefix_masked_zero"),
    ]

    global_rows = []
    for model_name, pred_col, prob_prefix in model_specs:
        metrics = aggregate_model_metrics(preds, pred_col, prob_prefix)
        row = {"model": model_name, "n_documents": len(preds), "n_patients": preds[PATIENT_COL].nunique()}
        for key, value in metrics.items():
            if key != "pred":
                row[key] = value
        global_rows.append(row)

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(aggregate_dir / "aggregate_metrics.csv", index=False)

    # ---------- MODEL-LEVEL BOOTSTRAP CIs ----------
    model_ci_parts = []
    for model_idx, (model_name, pred_col, prob_prefix) in enumerate(model_specs):
        model_ci_parts.append(
            patient_cluster_bootstrap_model_metrics(
                preds,
                model_name=model_name,
                prob_prefix=prob_prefix,
                n_boot=N_BOOTSTRAP,
                seed=BOOTSTRAP_SEED + 100 + model_idx,
            )
        )

    model_ci = pd.concat(model_ci_parts, ignore_index=True)
    model_ci.to_csv(
        aggregate_dir / "aggregate_metrics_with_ci.csv",
        index=False,
    )

    # ---------- PAIRED BOOTSTRAP ----------
    comparisons = []
    comparisons.append(patient_cluster_bootstrap_differences(preds, "fixed", "standard", seed=BOOTSTRAP_SEED))
    comparisons.append(patient_cluster_bootstrap_differences(preds, "fixed", "masked", seed=BOOTSTRAP_SEED + 1))
    comparisons.append(patient_cluster_bootstrap_differences(preds, "standard", "masked", seed=BOOTSTRAP_SEED + 2))
    paired = pd.concat(comparisons, ignore_index=True)
    paired.to_csv(aggregate_dir / "paired_patient_bootstrap_differences.csv", index=False)

    # ---------- PER-LOCALITY DESCRIPTIVE METRICS ----------
    per_loc_rows = []
    for locality in localities:
        d = preds[preds[GROUP_COL].astype(str) == locality].copy()
        for model_name, pred_col, prob_prefix in model_specs:
            metrics = aggregate_model_metrics(d, pred_col, prob_prefix)
            row = {
                "localita": locality,
                "model": model_name,
                "n_documents": len(d),
                "n_patients": d[PATIENT_COL].nunique(),
            }
            for key, value in metrics.items():
                if key != "pred":
                    row[key] = value
            per_loc_rows.append(row)

    per_loc = pd.DataFrame(per_loc_rows)
    per_loc.to_csv(aggregate_dir / "per_locality_metrics.csv", index=False)

    # ---------- PLOT: ACCURACY BY HELD-OUT LOCALITY ----------
    pivot = per_loc.pivot(index="localita", columns="model", values="accuracy")
    n_map = per_loc.drop_duplicates("localita").set_index("localita")["n_documents"]
    order = n_map.sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(order))
    offsets = [-0.20, 0.0, 0.20]

    for offset, (model_name, _, _) in zip(offsets, model_specs):
        ax.scatter(pivot[model_name].to_numpy(), y + offset, label=model_name, s=36)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{loc} (n={int(n_map.loc[loc])})" for loc in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy on held-out locality")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(aggregate_dir / "unseen_accuracy_by_locality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- PLOT: PAIRED DIFFERENCES VS FIXED ----------
    plot_df = paired[
        (paired["model_a"] == "fixed")
        & (paired["metric"].isin(["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]))
    ].copy()

    label_map = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro-F1",
        "macro_auc": "Macro-AUC",
    }
    model_label = {
        "standard": "Standard prefix",
        "masked": "10% zero-prefix masking",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    metrics_order = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]
    y_base = np.arange(len(metrics_order))

    for j, model_b in enumerate(["standard", "masked"]):
        temp = plot_df[plot_df["model_b"] == model_b].set_index("metric").loc[metrics_order]
        offset = -0.09 if j == 0 else 0.09
        x = temp["delta_b_minus_a"].to_numpy()
        xerr = np.vstack([
            x - temp["ci95_low"].to_numpy(),
            temp["ci95_high"].to_numpy() - x,
        ])
        ax.errorbar(
            x,
            y_base + offset,
            xerr=xerr,
            fmt="o",
            capsize=3,
            label=model_label[model_b],
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels([label_map[m] for m in metrics_order])
    ax.set_xlabel("Difference vs fixed-only (group-prefix minus fixed-only)")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(aggregate_dir / "unseen_paired_differences_vs_fixed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- PLOT: BRIER DIFFERENCE VS FIXED ----------
    brier_plot = paired[
        (paired["model_a"] == "fixed")
        & (paired["metric"] == "brier")
        & (paired["model_b"].isin(["standard", "masked"]))
    ].copy()

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = [model_label[m] for m in brier_plot["model_b"]]
    x = brier_plot["delta_b_minus_a"].to_numpy()
    y = np.arange(len(labels))
    xerr = np.vstack([
        x - brier_plot["ci95_low"].to_numpy(),
        brier_plot["ci95_high"].to_numpy() - x,
    ])
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Brier-score difference vs fixed-only (negative favours group-prefix)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(aggregate_dir / "unseen_brier_difference_vs_fixed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 100)
    print("AGGREGATE UNSEEN-LOCALITY RESULTS")
    print("=" * 100)
    print(global_df[[
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auc",
        "brier",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMODEL-LEVEL PATIENT-CLUSTER BOOTSTRAP 95% CIs")
    print(model_ci[[
        "model",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPAIRED PATIENT-CLUSTER BOOTSTRAP DIFFERENCES")
    print(paired[[
        "model_a",
        "model_b",
        "metric",
        "delta_b_minus_a",
        "ci95_low",
        "ci95_high",
    ]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\nPER-LOCALITY ACCURACY")
    print(
        per_loc.pivot(index="localita", columns="model", values="accuracy")
        .join(n_map.rename("n_documents"))
        .sort_values("n_documents", ascending=False)
        .to_string(float_format=lambda x: f"{x:.3f}")
    )

    print("\nSaved aggregate outputs in:", aggregate_dir)


# ============================================================
# 13. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, help="Run one held-out-locality fold (0..15)")
    group.add_argument("--aggregate", action="store_true", help="Aggregate all completed folds")
    group.add_argument("--list-folds", action="store_true", help="Print fold index -> locality mapping")
    args = parser.parse_args()

    train, test = load_data()
    localities = sorted(test[GROUP_COL].astype(str).unique().tolist())

    if args.list_folds:
        print("Fold mapping:")
        for i, loc in enumerate(localities):
            n_test = int((test[GROUP_COL].astype(str) == loc).sum())
            n_train = int((train[GROUP_COL].astype(str) == loc).sum())
            print(f"{i:02d}  {loc:<25s} train={n_train:4d}  test={n_test:3d}")
        return

    if args.aggregate:
        aggregate_results()
        return

    run_fold(args.fold)


if __name__ == "__main__":
    main()
