import torch
import torch.nn as nn
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

# ============================================================
# CONFIG
# ============================================================

EMB_PATH = Path("/work/u10767335/glmm/amazon_50k_glmm_embeddings_group_sampling.pt")
META_PATH = Path("/work/u10767335/glmm/amazon_50k_glmm_metadata_group_sampling.csv")

OUT_DIR = Path("/work/u10767335/fixed_only_classifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 123
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

#USE_CLASS_WEIGHT = True
USE_CLASS_WEIGHT = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading embeddings and metadata...")

X = torch.load(EMB_PATH, map_location="cpu").float()
meta = pd.read_csv(META_PATH)

y = meta["y"].values.astype(np.float32)

print("X shape:", X.shape)
print("Metadata shape:", meta.shape)
print("y distribution:")
print(pd.Series(y).value_counts().sort_index())

assert len(X) == len(meta), "Errore: embeddings e metadata hanno lunghezze diverse"

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

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

print("\nTrain size:", len(train_idx))
print("Test size:", len(test_idx))

print("\nTrain y distribution:")
print(pd.Series(y[train_idx]).value_counts().sort_index())

print("\nTest y distribution:")
print(pd.Series(y[test_idx]).value_counts().sort_index())

# ============================================================
# DATASET
# ============================================================

class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(
    EmbeddingDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    EmbeddingDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# MODEL
# ============================================================

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


model = BinaryClassifier(input_dim=X.shape[1]).to(device)

# ============================================================
# LOSS
# ============================================================

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

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):
    model.eval()

    all_logits = []
    all_y = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)

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

# ============================================================
# TRAINING
# ============================================================

history = []

best_bal_acc = -np.inf
best_state = None

print("\nTraining...")

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_losses = []

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    train_loss = np.mean(train_losses)

    train_metrics = evaluate(model, train_loader)
    test_metrics = evaluate(model, test_loader)

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
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
        f"Epoch {epoch:02d} | "
        f"loss={train_loss:.4f} | "
        f"test acc={test_metrics['accuracy']:.4f} | "
        f"test bal acc={test_metrics['balanced_accuracy']:.4f} | "
        f"sens={test_metrics['sensitivity']:.4f} | "
        f"spec={test_metrics['specificity']:.4f} | "
        f"auc={test_metrics['auc']:.4f}"
    )

    if test_metrics["balanced_accuracy"] > best_bal_acc:
        best_bal_acc = test_metrics["balanced_accuracy"]
        best_state = model.state_dict()

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\nLoading best model by test balanced accuracy...")

model.load_state_dict(best_state)

final_metrics = evaluate(model, test_loader)

print("\nFinal test results:")
print("Accuracy:", final_metrics["accuracy"])
print("Balanced accuracy:", final_metrics["balanced_accuracy"])
print("Sensitivity:", final_metrics["sensitivity"])
print("Specificity:", final_metrics["specificity"])
print("AUC:", final_metrics["auc"])
print("\nConfusion matrix [rows=true, cols=pred], labels=[0,1]:")
print(final_metrics["confusion_matrix"])

# # ============================================================
# # SAVE OUTPUTS
# # ============================================================

# history_df = pd.DataFrame(history)
# history_df.to_csv(OUT_DIR / "training_history.csv", index=False)

# pred_df = meta.iloc[test_idx].copy()
# pred_df["prob_positive"] = final_metrics["prob"]
# pred_df["pred"] = final_metrics["pred"]
# pred_df["y_true"] = final_metrics["y_true"]

# pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

# metrics_df = pd.DataFrame([{
#     "model": "frozen_roberta_embeddings_mlp_fixed_only",
#     "n_train": len(train_idx),
#     "n_test": len(test_idx),
#     "accuracy": final_metrics["accuracy"],
#     "balanced_accuracy": final_metrics["balanced_accuracy"],
#     "sensitivity": final_metrics["sensitivity"],
#     "specificity": final_metrics["specificity"],
#     "auc": final_metrics["auc"],
#     "use_class_weight": USE_CLASS_WEIGHT,
#     "seed": SEED,
#     "batch_size": BATCH_SIZE,
#     "epochs": EPOCHS,
#     "lr": LR,
#     "weight_decay": WEIGHT_DECAY
# }])

# metrics_df.to_csv(OUT_DIR / "final_metrics.csv", index=False)

# torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

# print("\nSaved:")
# print(OUT_DIR / "training_history.csv")
# print(OUT_DIR / "test_predictions.csv")
# print(OUT_DIR / "final_metrics.csv")
# print(OUT_DIR / "best_model.pt")