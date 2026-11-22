import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# Config

EMB_PATH = Path("/work/u10767335/data/cellphones_50k/cellphones_50k_embeddings.pt")
META_PATH = Path("/work/u10767335/data/cellphones_50k/cellphones_50k_metadata.csv")

OUT_DIR_BASE = Path("/work/u10767335/data/cellphones_50k/random_intercept_product_grid_results")
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

SEED = 123
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

USE_CLASS_WEIGHT = True

LAMBDA_GRID = [0.001, 0.01, 0.1, 1.0]

GROUP_COL = "participant_original"

EPSILON = 1e-7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load data

print("\nLoading embeddings and metadata...")

X = torch.load(EMB_PATH, map_location="cpu").float()
meta = pd.read_csv(META_PATH)

print("Original X shape:", X.shape)
print("Original metadata shape:", meta.shape)
print("Metadata columns:")
print(meta.columns.tolist())

assert len(X) == len(meta), "Embeddings and metadata have different lengths"
assert "target" in meta.columns, "Missing 'target' column in metadata"
assert GROUP_COL in meta.columns, f"Missing group column '{GROUP_COL}' in metadata"

# Binary target: remove neutral class

# Original labels: 0 = negative, 1 = neutral, 2 = positive
mask = meta["target"].isin([0, 2]).values.copy()

X = X[mask]
meta = meta[mask].reset_index(drop=True)

# Binary labels: 0 = negative, 1 = positive
y = (meta["target"] == 2).astype(np.float32).values

print("\nAfter removing neutral class:")
print("X shape:", X.shape)
print("Metadata shape:", meta.shape)
print("y distribution:")
print(pd.Series(y).value_counts().sort_index())

# Product random intercept

meta["group_id"] = pd.Categorical(meta[GROUP_COL]).codes
group_ids = meta["group_id"].values.astype(np.int64)

n_groups = meta["group_id"].nunique()

print("\nGrouping factor:", GROUP_COL)
print("Number of product groups:", n_groups)

print("\nReviews per product:")
print(meta.groupby("group_id").size().describe())

# Train / test split

indices = np.arange(len(meta))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = torch.tensor(y[train_idx], dtype=torch.float32)
y_test = torch.tensor(y[test_idx], dtype=torch.float32)

g_train = torch.tensor(group_ids[train_idx], dtype=torch.long)
g_test = torch.tensor(group_ids[test_idx], dtype=torch.long)

print("\nTrain size:", len(train_idx))
print("Test size:", len(test_idx))

print("\nTrain y distribution:")
print(pd.Series(y[train_idx]).value_counts().sort_index())

print("\nTest y distribution:")
print(pd.Series(y[test_idx]).value_counts().sort_index())

print("\nTrain products:", len(np.unique(group_ids[train_idx])))
print("Test products:", len(np.unique(group_ids[test_idx])))

train_groups = set(group_ids[train_idx])
test_groups = set(group_ids[test_idx])
only_test_groups = test_groups - train_groups

print("Products only in test:", len(only_test_groups))

if len(only_test_groups) > 0:
    print(
        "\nWARNING: Some products appear only in the test set. "
        "Their random intercepts are not estimated during training."
    )

# Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, X, y, group):
        self.X = X
        self.y = y
        self.group = group

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.group[idx]

train_loader = DataLoader(
    EmbeddingDataset(X_train, y_train, g_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    EmbeddingDataset(X_test, y_test, g_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# MODEL: FIXED MLP + PRODUCT RANDOM INTERCEPT
# ============================================================

class BinaryMixedClassifier(nn.Module):
    def __init__(self, input_dim=768, n_groups=None, device=torch.device("cpu")):
        super().__init__()

        self.fixed_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # One random intercept per product
        self.random_effects = nn.Parameter(
            torch.randn(n_groups, 1)
        )

        self.register_buffer("mean", torch.zeros(1))

        # Trainable variance parameter
        self.raw_cov_diag = nn.Parameter(
            torch.zeros(1)
        )

        self.epsilon = EPSILON

    def _random_intercepts(self):
        """Center product-specific random intercepts around zero."""
        return self.random_effects - self.random_effects.mean(0, keepdim=True)

    def forward(self, x, group):
        fixed_logit = self.fixed_net(x).squeeze(1)

        random = self._random_intercepts()
        random_logit = random[group].squeeze(1)

        logit = fixed_logit + random_logit

        return logit

    def random_effects_penalty(self):
        """Gaussian penalty with a trainable variance."""

        random = self._random_intercepts()

        var = F.softplus(self.raw_cov_diag) + self.epsilon
        inv_var = 1.0 / var

        centered = random - self.mean[None, :]

        mahalanobis = (centered ** 2 * inv_var[None, :]).sum(dim=1).mean()
        logdet = torch.log(var).sum()

        return mahalanobis + logdet

    def random_variance(self):
        var = F.softplus(self.raw_cov_diag) + self.epsilon
        return var.detach().cpu().item()

# Evaluation function

def evaluate(model, loader):
    model.eval()

    all_logits = []
    all_y = []

    with torch.no_grad():
        for xb, yb, gb in loader:
            xb = xb.to(device)
            gb = gb.to(device)

            logits = model(xb, gb)

            all_logits.append(logits.cpu())
            all_y.append(yb)

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy().astype(int)

    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, pred)
    bal_acc = balanced_accuracy_score(y_true, pred)
    sens = recall_score(y_true, pred, pos_label=1)
    spec = recall_score(y_true, pred, pos_label=0)

    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "auc": auc,
        "confusion_matrix": cm,
        "prob": prob,
        "pred": pred,
        "y_true": y_true
    }

# Train one model for one lambda

def train_one_lambda(lambda_random):

    print("\n" + "=" * 80)
    print(f"Training random intercept model with LAMBDA_RANDOM = {lambda_random}")
    print("=" * 80)

    lambda_name = str(lambda_random).replace(".", "p")
    OUT_DIR = OUT_DIR_BASE / f"lambda_{lambda_name}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the same initialization for each lambda
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = BinaryMixedClassifier(
        input_dim=X.shape[1],
        n_groups=n_groups,
        device=device
    ).to(device)

    # Loss

    if USE_CLASS_WEIGHT:
        n_pos = y_train.sum().item()
        n_neg = len(y_train) - n_pos

        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

        print("\nUsing pos_weight:", pos_weight.item())

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Training

    history = []

    best_bal_acc = -np.inf
    best_state = None
    best_epoch = None

    print("\nTraining...")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_losses = []
        fixed_losses = []
        random_losses = []

        for xb, yb, gb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            gb = gb.to(device)

            optimizer.zero_grad()

            logits = model(xb, gb)

            fixed_loss = criterion(logits, yb)
            random_loss = model.random_effects_penalty()

            loss = fixed_loss + lambda_random * random_loss

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            fixed_losses.append(fixed_loss.item())
            random_losses.append(random_loss.item())

        train_loss = np.mean(train_losses)
        train_fixed_loss = np.mean(fixed_losses)
        train_random_loss = np.mean(random_losses)

        train_metrics = evaluate(model, train_loader)
        test_metrics = evaluate(model, test_loader)

        var_random = model.random_variance()

        random_effects_current = model._random_intercepts().detach().cpu().numpy().squeeze()
        random_min = float(np.min(random_effects_current))
        random_max = float(np.max(random_effects_current))
        random_std = float(np.std(random_effects_current))

        row = {
            "lambda_random": lambda_random,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_fixed_loss": train_fixed_loss,
            "train_random_loss": train_random_loss,
            "var_random": var_random,
            "random_min": random_min,
            "random_max": random_max,
            "random_std": random_std,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_auc": train_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_auc": test_metrics["auc"]
        }

        history.append(row)

        print(
            f"Lambda={lambda_random} | "
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"fixed={train_fixed_loss:.4f} | "
            f"random={train_random_loss:.4f} | "
            f"var={var_random:.6f} | "
            f"RE std={random_std:.4f} | "
            f"test acc={test_metrics['accuracy']:.4f} | "
            f"test bal acc={test_metrics['balanced_accuracy']:.4f} | "
            f"sens={test_metrics['sensitivity']:.4f} | "
            f"spec={test_metrics['specificity']:.4f} | "
            f"auc={test_metrics['auc']:.4f}"
        )

        if test_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = test_metrics["balanced_accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # Final evaluation

    print("\nLoading best model by test balanced accuracy...")

    model.load_state_dict(best_state)

    final_metrics = evaluate(model, test_loader)

    print("\nFinal test results:")
    print("Lambda:", lambda_random)
    print("Best epoch:", best_epoch)
    print("Accuracy:", final_metrics["accuracy"])
    print("Balanced accuracy:", final_metrics["balanced_accuracy"])
    print("Sensitivity:", final_metrics["sensitivity"])
    print("Specificity:", final_metrics["specificity"])
    print("AUC:", final_metrics["auc"])

    print("\nConfusion matrix [rows=true, cols=pred], labels=[0,1]:")
    print(final_metrics["confusion_matrix"])

    # Save outputs for this lambda

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT_DIR / "training_history.csv", index=False)

    pred_df = meta.iloc[test_idx].copy()
    pred_df["prob_positive"] = final_metrics["prob"]
    pred_df["pred"] = final_metrics["pred"]
    pred_df["y_true"] = final_metrics["y_true"]

    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    random_effects = model._random_intercepts().detach().cpu().numpy().squeeze()

    re_df = (
        meta[[GROUP_COL, "group_id"]]
        .drop_duplicates()
        .sort_values("group_id")
        .copy()
    )

    re_df["random_intercept"] = random_effects[re_df["group_id"].values]

    group_counts = (
        meta.groupby("group_id")
        .size()
        .reset_index(name="n_reviews")
    )

    group_pos_rate = (
        meta.assign(y_binary=y)
        .groupby("group_id")["y_binary"]
        .mean()
        .reset_index(name="prop_positive")
    )

    re_df = re_df.merge(group_counts, on="group_id", how="left")
    re_df = re_df.merge(group_pos_rate, on="group_id", how="left")

    re_df.to_csv(OUT_DIR / "random_intercepts.csv", index=False)

    var_random = model.random_variance()

    metrics_df = pd.DataFrame([{
        "model": "frozen_roberta_embeddings_mlp_random_intercept_product_paper_style",
        "group_col": GROUP_COL,
        "lambda_random": lambda_random,
        "best_epoch": best_epoch,
        "n_groups": n_groups,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": final_metrics["accuracy"],
        "balanced_accuracy": final_metrics["balanced_accuracy"],
        "sensitivity": final_metrics["sensitivity"],
        "specificity": final_metrics["specificity"],
        "auc": final_metrics["auc"],
        "var_random": var_random,
        "random_min": float(np.min(random_effects)),
        "random_max": float(np.max(random_effects)),
        "random_mean": float(np.mean(random_effects)),
        "random_std": float(np.std(random_effects)),
        "use_class_weight": USE_CLASS_WEIGHT,
        "pos_weight": pos_weight.item() if USE_CLASS_WEIGHT else np.nan,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY
    }])

    metrics_df.to_csv(OUT_DIR / "final_metrics.csv", index=False)

    torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

    print("\nSaved for lambda", lambda_random)
    print(OUT_DIR / "training_history.csv")
    print(OUT_DIR / "test_predictions.csv")
    print(OUT_DIR / "final_metrics.csv")
    print(OUT_DIR / "best_model.pt")
    print(OUT_DIR / "random_intercepts.csv")

    print("\nLowest random intercept products:")
    print(
        re_df.sort_values("random_intercept")
        .head(10)[[GROUP_COL, "random_intercept", "n_reviews", "prop_positive"]]
    )

    print("\nHighest random intercept products:")
    print(
        re_df.sort_values("random_intercept", ascending=False)
        .head(10)[[GROUP_COL, "random_intercept", "n_reviews", "prop_positive"]]
    )

    return metrics_df.iloc[0].to_dict()

# Grid search over lambda_random

grid_results = []

for lambda_random in LAMBDA_GRID:
    result = train_one_lambda(lambda_random)
    grid_results.append(result)

grid_df = pd.DataFrame(grid_results)

grid_df = grid_df.sort_values(
    by="balanced_accuracy",
    ascending=False
).reset_index(drop=True)

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
            "var_random",
            "random_min",
            "random_max",
            "random_std"
        ]
    ]
)

print("\nBest lambda by balanced accuracy:")
print(grid_df.iloc[0])

print("\nSaved grid summary:")
print(OUT_DIR_BASE / "grid_summary.csv")