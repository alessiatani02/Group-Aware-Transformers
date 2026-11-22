import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

EMB_PATH = Path(
    "/work/u10767335/data/cellphones_50k/"
    "cellphones_50k_embeddings.pt"
)

META_PATH = Path(
    "/work/u10767335/data/cellphones_50k/"
    "cellphones_50k_metadata.csv"
)

# Original 3-class subset.
# We need this file because it still contains the text column,
# which is required to remove blank reviews.
DATA_PATH = Path(
    "/work/u10767335/data/cellphones_50k/"
    "cellphones_50k_3class_group_sampling.csv.gz"
)

OUT_DIR = Path(
    "/work/u10767335/data/cellphones_50k/"
    "fixed_only_results"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 123

BATCH_SIZE = 256
EPOCHS = 30

LR = 1e-3
WEIGHT_DECAY = 1e-4

EVALUATION_SIZE = 0.20

USE_CLASS_WEIGHT = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading embeddings and metadata...")

X = torch.load(
    EMB_PATH,
    map_location="cpu",
).float()

meta = pd.read_csv(
    META_PATH
)

full_data = pd.read_csv(
    DATA_PATH,
    compression="gzip",
)


print("\nOriginal data:")
print("Embeddings shape:", X.shape)
print("Metadata shape:", meta.shape)
print("Full dataset shape:", full_data.shape)


# ============================================================
# ALIGNMENT CHECKS
# ============================================================

assert len(X) == len(meta), (
    "Embeddings and metadata have different lengths."
)

assert len(meta) == len(full_data), (
    "Metadata and source dataset have different lengths."
)

assert "global_idx" in meta.columns
assert "global_idx" in full_data.columns

assert np.array_equal(
    meta["global_idx"].to_numpy(),
    full_data["global_idx"].to_numpy(),
), (
    "Metadata and source dataset are not aligned "
    "according to global_idx."
)

print("\nAlignment check passed.")


# ============================================================
# FINAL BINARY DATASET
# ============================================================

# Original target:
# 0 = negative
# 1 = neutral
# 2 = positive
#
# Final task:
# retain only negative and positive reviews,
# and remove missing or blank text.

binary_mask = (
    meta["target"]
    .isin([0, 2])
)

valid_text_mask = (
    full_data["text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

final_mask = (
    binary_mask
    &
    valid_text_mask
).to_numpy()


X = X[
    final_mask
]

meta = (
    meta.loc[
        final_mask
    ]
    .reset_index(
        drop=True
    )
)

full_data = (
    full_data.loc[
        final_mask
    ]
    .reset_index(
        drop=True
    )
)


# 0 = negative
# 1 = positive
y = (
    meta["target"]
    ==
    2
).astype(
    np.float32
).to_numpy()


print("\n" + "=" * 80)
print("FINAL BINARY DATASET")
print("=" * 80)

print(
    "X shape:",
    X.shape,
)

print(
    "Metadata shape:",
    meta.shape,
)

print(
    "\nTarget distribution:"
)

print(
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print(
    "\nNumber of groups:",
    meta[
        "participant_original"
    ].nunique()
)


assert len(X) == len(meta)

assert len(meta) == 46307, (
    f"Expected 46,307 reviews, found {len(meta)}."
)


# ============================================================
# TRAIN / HELD-OUT EVALUATION SPLIT
# ============================================================

indices = np.arange(
    len(meta)
)

train_idx, evaluation_idx = train_test_split(
    indices,
    test_size=EVALUATION_SIZE,
    random_state=SEED,
    stratify=y,
)


X_train = X[
    train_idx
]

X_evaluation = X[
    evaluation_idx
]


y_train = torch.tensor(
    y[train_idx],
    dtype=torch.float32,
)

y_evaluation = torch.tensor(
    y[evaluation_idx],
    dtype=torch.float32,
)


print("\n" + "=" * 80)
print("TRAIN / HELD-OUT EVALUATION SPLIT")
print("=" * 80)

print(
    "Train size:",
    len(train_idx),
)

print(
    "Evaluation size:",
    len(evaluation_idx),
)


print(
    "\nTrain target distribution:"
)

print(
    pd.Series(
        y[train_idx]
    )
    .value_counts()
    .sort_index()
)


print(
    "\nEvaluation target distribution:"
)

print(
    pd.Series(
        y[evaluation_idx]
    )
    .value_counts()
    .sort_index()
)


# ============================================================
# DATASET
# ============================================================

class EmbeddingDataset(Dataset):

    def __init__(
        self,
        X,
        y,
    ):

        self.X = X
        self.y = y

    def __len__(
        self,
    ):

        return len(
            self.y
        )

    def __getitem__(
        self,
        idx,
    ):

        return (
            self.X[idx],
            self.y[idx],
        )


train_loader = DataLoader(
    EmbeddingDataset(
        X_train,
        y_train,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


evaluation_loader = DataLoader(
    EmbeddingDataset(
        X_evaluation,
        y_evaluation,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

class BinaryClassifier(nn.Module):

    def __init__(
        self,
        input_dim=768,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.net(x)
            .squeeze(1)
        )


model = BinaryClassifier(
    input_dim=X.shape[1]
).to(
    device
)


print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ),
)


# ============================================================
# LOSS
# ============================================================

if USE_CLASS_WEIGHT:

    n_pos = (
        y_train
        .sum()
        .item()
    )

    n_neg = (
        len(y_train)
        -
        n_pos
    )

    pos_weight_value = (
        n_neg
        /
        n_pos
    )

    pos_weight = torch.tensor(
        [
            pos_weight_value
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "\nUsing pos_weight:",
        pos_weight.item(),
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

else:

    pos_weight_value = np.nan

    criterion = (
        nn.BCEWithLogitsLoss()
    )


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
):

    model.eval()

    all_logits = []
    all_y = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(
                device
            )

            logits = model(
                xb
            )

            all_logits.append(
                logits.cpu()
            )

            all_y.append(
                yb
            )


    logits = (
        torch.cat(
            all_logits
        )
        .numpy()
    )

    y_true = (
        torch.cat(
            all_y
        )
        .numpy()
        .astype(int)
    )


    prob = (
        1
        /
        (
            1
            +
            np.exp(
                -logits
            )
        )
    )

    pred = (
        prob
        >=
        0.5
    ).astype(int)


    acc = accuracy_score(
        y_true,
        pred,
    )

    bal_acc = (
        balanced_accuracy_score(
            y_true,
            pred,
        )
    )

    sens = recall_score(
        y_true,
        pred,
        pos_label=1,
    )

    spec = recall_score(
        y_true,
        pred,
        pos_label=0,
    )

    try:

        auc = roc_auc_score(
            y_true,
            prob,
        )

    except ValueError:

        auc = np.nan


    cm = confusion_matrix(
        y_true,
        pred,
        labels=[
            0,
            1,
        ],
    )


    return {

        "accuracy":
            acc,

        "balanced_accuracy":
            bal_acc,

        "sensitivity":
            sens,

        "specificity":
            spec,

        "auc":
            auc,

        "confusion_matrix":
            cm,

        "prob":
            prob,

        "pred":
            pred,

        "y_true":
            y_true,
    }


# ============================================================
# TRAINING
# ============================================================

history = []

best_bal_acc = (
    -np.inf
)

best_state = None
best_epoch = None


print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    train_losses = []


    for xb, yb in train_loader:

        xb = xb.to(
            device
        )

        yb = yb.to(
            device
        )


        optimizer.zero_grad()


        logits = model(
            xb
        )

        loss = criterion(
            logits,
            yb,
        )


        loss.backward()

        optimizer.step()


        train_losses.append(
            loss.item()
        )


    train_loss = np.mean(
        train_losses
    )


    train_metrics = evaluate(
        model,
        train_loader,
    )

    evaluation_metrics = evaluate(
        model,
        evaluation_loader,
    )


    row = {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_metrics[
                "accuracy"
            ],

        "train_balanced_accuracy":
            train_metrics[
                "balanced_accuracy"
            ],

        "train_auc":
            train_metrics[
                "auc"
            ],

        "evaluation_accuracy":
            evaluation_metrics[
                "accuracy"
            ],

        "evaluation_balanced_accuracy":
            evaluation_metrics[
                "balanced_accuracy"
            ],

        "evaluation_sensitivity":
            evaluation_metrics[
                "sensitivity"
            ],

        "evaluation_specificity":
            evaluation_metrics[
                "specificity"
            ],

        "evaluation_auc":
            evaluation_metrics[
                "auc"
            ],
    }


    history.append(
        row
    )


    print(
        f"Epoch {epoch:02d} | "
        f"loss={train_loss:.4f} | "
        f"eval acc={evaluation_metrics['accuracy']:.4f} | "
        f"eval bal acc={evaluation_metrics['balanced_accuracy']:.4f} | "
        f"sens={evaluation_metrics['sensitivity']:.4f} | "
        f"spec={evaluation_metrics['specificity']:.4f} | "
        f"auc={evaluation_metrics['auc']:.4f}"
    )


    if (
        evaluation_metrics[
            "balanced_accuracy"
        ]
        >
        best_bal_acc
    ):

        best_bal_acc = (
            evaluation_metrics[
                "balanced_accuracy"
            ]
        )

        best_epoch = epoch

        
        best_state = copy.deepcopy(
            model.state_dict()
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

print(
    "\nLoading best model by "
    "held-out balanced accuracy..."
)

print(
    "Best epoch:",
    best_epoch,
)

model.load_state_dict(
    best_state
)


# ============================================================
# FINAL HELD-OUT EVALUATION
# ============================================================

final_metrics = evaluate(
    model,
    evaluation_loader,
)


print("\n" + "=" * 80)
print("FINAL HELD-OUT EVALUATION")
print("=" * 80)

print(
    "Best epoch:",
    best_epoch,
)

print(
    "Accuracy:",
    final_metrics[
        "accuracy"
    ],
)

print(
    "Balanced accuracy:",
    final_metrics[
        "balanced_accuracy"
    ],
)

print(
    "Sensitivity:",
    final_metrics[
        "sensitivity"
    ],
)

print(
    "Specificity:",
    final_metrics[
        "specificity"
    ],
)

print(
    "AUC:",
    final_metrics[
        "auc"
    ],
)

print(
    "\nConfusion matrix "
    "[rows=true, cols=pred], "
    "labels=[0,1]:"
)

print(
    final_metrics[
        "confusion_matrix"
    ]
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUT_DIR
    /
    "training_history.csv",
    index=False,
)


# ------------------------------------------------------------
# Evaluation predictions
# ------------------------------------------------------------

pred_df = (
    meta.iloc[
        evaluation_idx
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

pred_df[
    "prob_positive"
] = final_metrics[
    "prob"
]

pred_df[
    "pred"
] = final_metrics[
    "pred"
]

pred_df[
    "y_true"
] = final_metrics[
    "y_true"
]


pred_df.to_csv(
    OUT_DIR
    /
    "evaluation_predictions.csv",
    index=False,
)


# ------------------------------------------------------------
# Final metrics
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    [
        {
            "model":
                "fixed_only_neural_model",

            "best_epoch":
                best_epoch,

            "n_total":
                len(meta),

            "n_train":
                len(train_idx),

            "n_evaluation":
                len(evaluation_idx),

            "accuracy":
                final_metrics[
                    "accuracy"
                ],

            "balanced_accuracy":
                final_metrics[
                    "balanced_accuracy"
                ],

            "sensitivity":
                final_metrics[
                    "sensitivity"
                ],

            "specificity":
                final_metrics[
                    "specificity"
                ],

            "auc":
                final_metrics[
                    "auc"
                ],

            "use_class_weight":
                USE_CLASS_WEIGHT,

            "pos_weight":
                pos_weight_value,

            "seed":
                SEED,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "lr":
                LR,

            "weight_decay":
                WEIGHT_DECAY,
        }
    ]
)


metrics_df.to_csv(
    OUT_DIR
    /
    "final_metrics.csv",
    index=False,
)


# ------------------------------------------------------------
# Save best model
# ------------------------------------------------------------

torch.save(
    best_state,
    OUT_DIR
    /
    "best_model.pt",
)


print("\n" + "=" * 80)
print("SAVED OUTPUTS")
print("=" * 80)

print(
    OUT_DIR
    /
    "training_history.csv"
)

print(
    OUT_DIR
    /
    "evaluation_predictions.csv"
)

print(
    OUT_DIR
    /
    "final_metrics.csv"
)

print(
    OUT_DIR
    /
    "best_model.pt"
)

print("\nDONE.")