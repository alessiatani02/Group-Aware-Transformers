import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from transformers import PreTrainedTokenizerFast, AutoModel


# ============================================================
# 1. CONFIG
# ============================================================
TRAIN_PATH = Path(
    "/work/u10767335/data/breath/breath_train_4class_preprocessed512_nodiag.csv.gz"
)

TEST_PATH = Path(
    "/work/u10767335/data/breath/breath_test_4class_preprocessed512_nodiag.csv.gz"
)
OUT_DIR = Path(
    "/work/u10767335/data/breath/fixed_only_umberto_preprocessed512_nodiag_4class_results"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
MODEL_REVISION = "e934c95baeee3f4e613d426d044e52baa82300df"
TEXT_COL = "testo_512"
TARGET_COL = "diagnosi"
GROUP_COL = "localita"      # used only to reproduce the same validation split
PATIENT_COL = "pedianet_id"
# Same explicit class order used in the group-prefix model.
CLASS_NAMES = [
    "bronchiolite",
    "bronchite",
    "laringite",
    "wheezing",
]
LABEL_TO_ID = {label: i for i, label in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
NUM_CLASSES = len(CLASS_NAMES)
# Same setup as the group-prefix experiment.
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 5
LR_CLASSIFIER = 1e-3
LR_LAST_LAYER = 2e-5
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHT = True
N_VAL_SPLITS = 5
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# 3. LOAD PRE-SPLIT BREATH DATA
# ============================================================
train_full = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
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
            f"Missing columns in {name}: {sorted(missing)}"
        )
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    if df[TEXT_COL].str.strip().eq("").any():
        raise ValueError(
            f"Empty texts found in {name}. "
            "The preprocessing pipeline should have removed them."
        )
    unknown_labels = set(df[TARGET_COL].unique()) - set(CLASS_NAMES)
    if unknown_labels:
        raise ValueError(
            f"Unexpected labels in {name}: {sorted(unknown_labels)}"
        )
# Verify that the fixed outer split is patient-disjoint.
patient_overlap = (
    set(train_full[PATIENT_COL])
    & set(test_df[PATIENT_COL])
)
if patient_overlap:
    raise ValueError(
        f"Patient leakage: {len(patient_overlap)} patients "
        "appear in both train and test."
    )
print("Patient overlap train/test:", len(patient_overlap))


# ============================================================
# 4. LABEL IDS
# ============================================================
train_full["label_id"] = (
    train_full[TARGET_COL]
    .map(LABEL_TO_ID)
    .astype(np.int64)
)
test_df["label_id"] = (
    test_df[TARGET_COL]
    .map(LABEL_TO_ID)
    .astype(np.int64)
)
print("\nClass mapping:", LABEL_TO_ID)
print("\nTrain target distribution:")
print(train_full[TARGET_COL].value_counts())
print("\nTest target distribution:")
print(test_df[TARGET_COL].value_counts())


# ============================================================
# 5. INNER TRAIN / VALIDATION SPLIT BY PATIENT
# ============================================================
def make_inner_split(df):
    """
    Reproduce the same patient-disjoint validation strategy used
    in the group-prefix script.
    We require:
      - all four classes in fit and validation;
      - every locality to remain represented in the fit subset.
    GROUP_COL is used only for split comparability. The fixed-only
    model never receives locality as an input feature.
    """
    all_groups = set(df[GROUP_COL].unique())
    all_classes = set(range(NUM_CLASSES))
    for random_state in range(SEED, SEED + 50):
        splitter = StratifiedGroupKFold(
            n_splits=N_VAL_SPLITS,
            shuffle=True,
            random_state=random_state,
        )
        for fold, (fit_idx, val_idx) in enumerate(
            splitter.split(
                df,
                y=df["label_id"],
                groups=df[PATIENT_COL],
            )
        ):
            fit_df = df.iloc[fit_idx]
            val_df = df.iloc[val_idx]
            fit_groups = set(fit_df[GROUP_COL].unique())
            fit_classes = set(fit_df["label_id"].unique())
            val_classes = set(val_df["label_id"].unique())
            if (
                fit_groups == all_groups
                and fit_classes == all_classes
                and val_classes == all_classes
            ):
                return (
                    np.asarray(fit_idx),
                    np.asarray(val_idx),
                    random_state,
                    fold,
                )
    raise RuntimeError(
        "Could not find a suitable patient-disjoint "
        "train/validation split."
    )
fit_idx, val_idx, val_seed, val_fold = make_inner_split(train_full)
fit_df = train_full.iloc[fit_idx].copy().reset_index(drop=True)
val_df = train_full.iloc[val_idx].copy().reset_index(drop=True)
test_df = test_df.copy().reset_index(drop=True)
inner_overlap = set(fit_df[PATIENT_COL]) & set(val_df[PATIENT_COL])
print("\nInner split random_state:", val_seed)
print("Inner split fold:", val_fold)
print("Fit shape:", fit_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)
print("Patient overlap fit/validation:", len(inner_overlap))
print("Fit localities:", fit_df[GROUP_COL].nunique())
print("Validation localities:", val_df[GROUP_COL].nunique())
print("Test localities:", test_df[GROUP_COL].nunique())


# ============================================================
# 6. TOKENIZER + DATASET
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
class ClinicalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            row[TEXT_COL],
            int(row["label_id"]),
            int(idx),
        )
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
        torch.tensor(
            [row[1] for row in batch],
            dtype=torch.long,
        ),
        torch.tensor(
            [row[2] for row in batch],
            dtype=torch.long,
        ),
    )
fit_loader = DataLoader(
    ClinicalDataset(fit_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
full_train_loader = DataLoader(
    ClinicalDataset(train_full),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
    ClinicalDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
    ClinicalDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# ============================================================
# 7. FIXED-ONLY MODEL
# ============================================================
class FixedOnlyClassifier(nn.Module):
    """
    Text-only baseline with minimal domain adaptation:
        text -> UmBERTo
             -> first 11 layers frozen
             -> last transformer layer trainable
             -> mean pooling over original text tokens
             -> MLP 768 -> 128 -> 4 logits
    No locality/group information is used by the model.
    """
    def __init__(self, n_classes):
        super().__init__()
        # UmBERTo is a RoBERTa/CamemBERT-style encoder pretrained
        # specifically on large-scale Italian text.
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            use_safetensors=True,
        )
        self.hidden_size = self.encoder.config.hidden_size
        # Freeze the whole encoder first.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        # Unfreeze only the last transformer layer.
        for parameter in self.encoder.encoder.layer[-1].parameters():
            parameter.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
    def train(self, mode=True):
        super().train(mode)
        # Keep all frozen parts of UmBERTo in eval mode, but allow the
        # last transformer layer to behave as a trainable layer.
        self.encoder.eval()
        self.encoder.encoder.layer[-1].train(mode)
        return self
    @staticmethod
    def mean_pool(hidden, attention_mask):
        mask = (
            attention_mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )
        return (
            (hidden * mask).sum(1)
            / mask.sum(1).clamp_min(1.0)
        )
    def forward(self, input_ids, attention_mask):
        # Do NOT wrap the encoder in torch.no_grad(): the last transformer
        # layer is trainable and must receive gradients. Frozen parameters
        # in the preceding layers still remain unchanged.
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        pooled = self.mean_pool(
            hidden,
            attention_mask,
        )
        logits = self.classifier(pooled)
        return logits


# ============================================================
# 8. MULTICLASS METRICS
# ============================================================
def softmax_numpy(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
def calculate_metrics(y_true, logits):
    prob = softmax_numpy(logits)
    pred = np.argmax(prob, axis=1)
    try:
        auc_macro_ovr = roc_auc_score(
            y_true,
            prob,
            multi_class="ovr",
            average="macro",
            labels=np.arange(NUM_CLASSES),
        )
    except ValueError:
        auc_macro_ovr = np.nan
    try:
        auc_weighted_ovr = roc_auc_score(
            y_true,
            prob,
            multi_class="ovr",
            average="weighted",
            labels=np.arange(NUM_CLASSES),
        )
    except ValueError:
        auc_weighted_ovr = np.nan
    recalls = recall_score(
        y_true,
        pred,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )
    result = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            pred,
        ),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            pred,
            average="weighted",
            zero_division=0,
        ),
        "auc_macro_ovr": auc_macro_ovr,
        "auc_weighted_ovr": auc_weighted_ovr,
        "confusion_matrix": confusion_matrix(
            y_true,
            pred,
            labels=np.arange(NUM_CLASSES),
        ),
        "logits": logits,
        "prob": prob,
        "pred": pred,
        "y_true": y_true,
    }
    for class_id, class_name in ID_TO_LABEL.items():
        result[f"recall_{class_name}"] = recalls[class_id]
    return result


# ============================================================
# 9. EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits_all = []
    labels_all = []
    indices_all = []
    for (
        input_ids,
        attention_mask,
        labels,
        original_idx,
    ) in loader:
        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True,
        )
        attention_mask = attention_mask.to(
            DEVICE,
            non_blocking=True,
        )
        logits = model(
            input_ids,
            attention_mask,
        )
        logits_all.append(logits.cpu())
        labels_all.append(labels)
        indices_all.append(original_idx)
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy().astype(int)
    original_indices = (
        torch.cat(indices_all).numpy().astype(int)
    )
    result = calculate_metrics(labels, logits)
    result["original_idx"] = original_indices
    return result


# ============================================================
# 10. CLASS WEIGHTS
# ============================================================
def make_class_weights(df):
    labels = df["label_id"].to_numpy(dtype=np.int64)
    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    )
    if np.any(counts == 0):
        raise ValueError(
            f"At least one class is absent from training set: {counts}"
        )
    weights = (
        len(labels)
        / (NUM_CLASSES * counts.astype(np.float64))
    )
    return counts, weights


# ============================================================
# 11. TRAIN ON INNER FIT SET AND CHOOSE BEST EPOCH
# ============================================================
print("\n" + "=" * 80)
print("FIXED-ONLY BREATH 4-CLASS - UmBERTo, LAST LAYER TRAINABLE")
print("=" * 80)
set_seed(SEED)
model = FixedOnlyClassifier(
    n_classes=NUM_CLASSES,
).to(DEVICE)
trainable_parameters = [
    p for p in model.parameters()
    if p.requires_grad
]
n_trainable = sum(
    p.numel() for p in trainable_parameters
)
print("Trainable parameters:", f"{n_trainable:,}")
class_counts, class_weights = make_class_weights(fit_df)
print("Fit class counts:")
for class_id, class_name in ID_TO_LABEL.items():
    print(
        f"  {class_name}: {class_counts[class_id]}"
    )
if USE_CLASS_WEIGHT:
    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )
    print("Class weights:", class_weights)
else:
    criterion = nn.CrossEntropyLoss()
    print("Class weights: disabled")
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.classifier.parameters(),
            "lr": LR_CLASSIFIER,
        },
        {
            "params": model.encoder.encoder.layer[-1].parameters(),
            "lr": LR_LAST_LAYER,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)
history = []
best_balanced_accuracy = -np.inf
best_epoch = None
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for (
        input_ids,
        attention_mask,
        labels,
        _,
    ) in fit_loader:
        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True,
        )
        attention_mask = attention_mask.to(
            DEVICE,
            non_blocking=True,
        )
        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids,
            attention_mask,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = float(np.mean(train_losses))
    # Model selection uses validation only. The external test set
    # is not touched during this phase.
    val_metrics = evaluate(
        model,
        val_loader,
    )
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics[
            "balanced_accuracy"
        ],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "val_auc_macro_ovr": val_metrics[
            "auc_macro_ovr"
        ],
        "val_auc_weighted_ovr": val_metrics[
            "auc_weighted_ovr"
        ],
    }
    for class_name in CLASS_NAMES:
        row[f"val_recall_{class_name}"] = (
            val_metrics[f"recall_{class_name}"]
        )
    history.append(row)
    print(
        f"Epoch {epoch:02d} | "
        f"loss={train_loss:.4f} | "
        f"val_acc={row['val_accuracy']:.4f} | "
        f"val_bal_acc={row['val_balanced_accuracy']:.4f} | "
        f"val_f1={row['val_macro_f1']:.4f} | "
        f"val_auc={row['val_auc_macro_ovr']:.4f}"
    )
    if (
        val_metrics["balanced_accuracy"]
        > best_balanced_accuracy
    ):
        best_balanced_accuracy = (
            val_metrics["balanced_accuracy"]
        )
        best_epoch = epoch
print("\nBEST VALIDATION MODEL")
print("Best epoch:", best_epoch)
print(
    "Best validation balanced accuracy:",
    best_balanced_accuracy,
)


# ============================================================
# 12. RETRAIN ON FULL SAVED TRAINING SET
# ============================================================
#
# The validation split was used only to select best_epoch.
# We now restart from the same seed and train a fresh fixed-only
# model on all 2,162 saved training documents for best_epoch
# epochs. The external test set is still untouched.
print(
    "\nRetraining final fixed-only model on the full saved "
    f"training set for {best_epoch} epoch(s)..."
)
set_seed(SEED)
final_model = FixedOnlyClassifier(
    n_classes=NUM_CLASSES,
).to(DEVICE)
final_trainable_parameters = [
    p for p in final_model.parameters()
    if p.requires_grad
]
full_class_counts, full_class_weights = make_class_weights(
    train_full
)
if USE_CLASS_WEIGHT:
    final_weight_tensor = torch.tensor(
        full_class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )
    final_criterion = nn.CrossEntropyLoss(
        weight=final_weight_tensor
    )
else:
    final_criterion = nn.CrossEntropyLoss()
final_optimizer = torch.optim.AdamW(
    [
        {
            "params": final_model.classifier.parameters(),
            "lr": LR_CLASSIFIER,
        },
        {
            "params": final_model.encoder.encoder.layer[-1].parameters(),
            "lr": LR_LAST_LAYER,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)
final_train_history = []
for final_epoch in range(1, best_epoch + 1):
    final_model.train()
    epoch_losses = []
    for (
        input_ids,
        attention_mask,
        labels,
        _,
    ) in full_train_loader:
        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True,
        )
        attention_mask = attention_mask.to(
            DEVICE,
            non_blocking=True,
        )
        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )
        final_optimizer.zero_grad(set_to_none=True)
        logits = final_model(
            input_ids,
            attention_mask,
        )
        loss = final_criterion(
            logits,
            labels,
        )
        loss.backward()
        final_optimizer.step()
        epoch_losses.append(loss.item())
    retrain_row = {
        "epoch": final_epoch,
        "train_loss": float(np.mean(epoch_losses)),
    }
    final_train_history.append(retrain_row)
    print(
        f"Final train epoch {final_epoch:02d} | "
        f"loss={retrain_row['train_loss']:.4f}"
    )


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================
#
# The external test set is evaluated exactly once, after all
# model-selection decisions have been made.
final_metrics = evaluate(
    final_model,
    test_loader,
)
print("\nFINAL TEST RESULTS")
print("Accuracy:", final_metrics["accuracy"])
print(
    "Balanced accuracy:",
    final_metrics["balanced_accuracy"],
)
print("Macro F1:", final_metrics["macro_f1"])
print("Weighted F1:", final_metrics["weighted_f1"])
print(
    "Macro OVR AUC:",
    final_metrics["auc_macro_ovr"],
)
print(
    "Weighted OVR AUC:",
    final_metrics["auc_weighted_ovr"],
)
for class_name in CLASS_NAMES:
    print(
        f"Recall {class_name}: ",
        final_metrics[f"recall_{class_name}"],
    )
print("\nConfusion matrix [rows=true, cols=pred]:")
print(final_metrics["confusion_matrix"])
print("Class order:", CLASS_NAMES)


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================
pd.DataFrame(history).to_csv(
    OUT_DIR / "training_history.csv",
    index=False,
)
pd.DataFrame(final_train_history).to_csv(
    OUT_DIR / "final_retrain_history.csv",
    index=False,
)
pred_df = (
    test_df
    .iloc[final_metrics["original_idx"]]
    .copy()
    .reset_index(drop=True)
)
pred_df["y_true_id"] = final_metrics["y_true"]
pred_df["pred_id"] = final_metrics["pred"]
pred_df["pred_diagnosi"] = [
    ID_TO_LABEL[int(i)]
    for i in final_metrics["pred"]
]
for class_id, class_name in ID_TO_LABEL.items():
    pred_df[f"prob_{class_name}"] = (
        final_metrics["prob"][:, class_id]
    )
    pred_df[f"logit_{class_name}"] = (
        final_metrics["logits"][:, class_id]
    )
pred_df.to_csv(
    OUT_DIR / "test_predictions.csv",
    index=False,
)
metrics = {
    "model": "umberto_last_layer_trainable_fixed_only_multiclass",
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "target_col": TARGET_COL,
    "n_classes": NUM_CLASSES,
    "n_fit": len(fit_df),
    "n_validation": len(val_df),
    "n_train_full": len(train_full),
    "n_test": len(test_df),
    "best_epoch": best_epoch,
    "accuracy": final_metrics["accuracy"],
    "balanced_accuracy": final_metrics[
        "balanced_accuracy"
    ],
    "macro_f1": final_metrics["macro_f1"],
    "weighted_f1": final_metrics["weighted_f1"],
    "auc_macro_ovr": final_metrics[
        "auc_macro_ovr"
    ],
    "auc_weighted_ovr": final_metrics[
        "auc_weighted_ovr"
    ],
    "use_class_weight": USE_CLASS_WEIGHT,
    "batch_size": BATCH_SIZE,
    "epochs_max": EPOCHS,
    "lr_classifier": LR_CLASSIFIER,
    "lr_last_layer": LR_LAST_LAYER,
    "weight_decay": WEIGHT_DECAY,
    "max_length": MAX_LENGTH,
    "n_trainable_parameters": n_trainable,
    "validation_random_state": val_seed,
    "validation_fold": val_fold,
    "seed": SEED,
}
for class_name in CLASS_NAMES:
    metrics[f"recall_{class_name}"] = (
        final_metrics[f"recall_{class_name}"]
    )
pd.DataFrame([metrics]).to_csv(
    OUT_DIR / "final_metrics.csv",
    index=False,
)
# Save the trainable parts: classifier and last UmBERTo transformer layer.
torch.save(
    {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "class_names": CLASS_NAMES,
        "classifier_state_dict": (
            final_model.classifier.state_dict()
        ),
        "last_layer_state_dict": (
            final_model.encoder.encoder.layer[-1].state_dict()
        ),
        "best_epoch": best_epoch,
        "max_length": MAX_LENGTH,
    },
    OUT_DIR / "best_model.pt",
)
config = {
    "train_path": str(TRAIN_PATH),
    "test_path": str(TEST_PATH),
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "text_col": TEXT_COL,
    "target_col": TARGET_COL,
    "patient_col": PATIENT_COL,
    "class_names": CLASS_NAMES,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs_max": EPOCHS,
    "best_epoch": best_epoch,
    "lr_classifier": LR_CLASSIFIER,
    "lr_last_layer": LR_LAST_LAYER,
    "weight_decay": WEIGHT_DECAY,
    "use_class_weight": USE_CLASS_WEIGHT,
    "validation_random_state": val_seed,
    "validation_fold": val_fold,
}
with open(
    OUT_DIR / "config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False,
    )
print("\nSaved:")
print(OUT_DIR / "training_history.csv")
print(OUT_DIR / "final_retrain_history.csv")
print(OUT_DIR / "test_predictions.csv")
print(OUT_DIR / "final_metrics.csv")
print(OUT_DIR / "best_model.pt")
print(OUT_DIR / "config.json")
