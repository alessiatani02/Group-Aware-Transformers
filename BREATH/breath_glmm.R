# ============================================================
# BREATH / PEDIANET



# ============================================================
# 1. PACCHETTI R
# ============================================================

packages <- c(
  "data.table",
  "mclogit",
  "reticulate",
  "R.utils"
)

missing_packages <- packages[
  !sapply(packages, requireNamespace, quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(
    missing_packages,
    repos = "https://cloud.r-project.org"
  )
}

library(data.table)
library(mclogit)
library(reticulate)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED <- 42
K <- 30
N_BOOT <- 2000

CLASSES <- c(
  "bronchiolite",
  "bronchite",
  "laringite",
  "wheezing"
)

set.seed(SEED)


# ============================================================
# 3. SELEZIONA TRAIN E TEST
# ============================================================

cat("\nSeleziona il TRAIN:\n")
TRAIN_PATH <- file.choose()

cat("\nSeleziona il TEST:\n")
TEST_PATH <- file.choose()


# Cartella risultati creata automaticamente
OUT_DIR <- file.path(
  dirname(TRAIN_PATH),
  "classical_multinomial_glmm"
)

dir.create(
  OUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)


# File cache embeddings
EMBEDDING_CACHE <- file.path(
  OUT_DIR,
  "breath_umberto_embeddings.rds"
)

RESULT_FILE <- file.path(
  OUT_DIR,
  "breath_GLMM_results.csv"
)

PRED_FILE <- file.path(
  OUT_DIR,
  "breath_GLMM_test_predictions.csv"
)

MODEL_FILE <- file.path(
  OUT_DIR,
  "breath_GLMM_model.rds"
)


# ============================================================
# 4. CREA GLI EMBEDDINGS SOLO SE NON ESISTONO
# ============================================================

if (!file.exists(EMBEDDING_CACHE)) {
  
  cat("\n========================================\n")
  cat("CREAZIONE EMBEDDINGS UMBERTO\n")
  cat("========================================\n")
  
  
  # ----------------------------------------------------------
  # Leggi i dataset
  # ----------------------------------------------------------
  
  train_df <- fread(TRAIN_PATH)
  test_df  <- fread(TEST_PATH)
  
  
  cat("\nTrain documents:", nrow(train_df), "\n")
  cat("Test documents :", nrow(test_df), "\n")
  
  
  if (nrow(train_df) != 2162) {
    stop(
      paste(
        "Il train dovrebbe avere 2162 righe, ma ne ha",
        nrow(train_df)
      )
    )
  }
  
  
  if (nrow(test_df) != 541) {
    stop(
      paste(
        "Il test dovrebbe avere 541 righe, ma ne ha",
        nrow(test_df)
      )
    )
  }
  
  
  # ----------------------------------------------------------
  # Controlla colonne
  # ----------------------------------------------------------
  
  required_cols <- c(
    "testo_512",
    "diagnosi",
    "localita",
    "pedianet_id"
  )
  
  
  if (!all(required_cols %in% names(train_df))) {
    stop(
      paste(
        "Colonne TRAIN:",
        paste(names(train_df), collapse = ", ")
      )
    )
  }
  
  
  if (!all(required_cols %in% names(test_df))) {
    stop(
      paste(
        "Colonne TEST:",
        paste(names(test_df), collapse = ", ")
      )
    )
  }
  
  
  # ----------------------------------------------------------
  # Controlla assenza overlap pazienti
  # ----------------------------------------------------------
  
  overlap <- intersect(
    unique(as.character(train_df$pedianet_id)),
    unique(as.character(test_df$pedianet_id))
  )
  
  
  cat(
    "Patient overlap train/test:",
    length(overlap),
    "\n"
  )
  
  
  if (length(overlap) != 0) {
    stop("Ci sono pazienti sia nel train sia nel test.")
  }
  
  
  # ----------------------------------------------------------
  # Identifica split
  # ----------------------------------------------------------
  
  train_df[, split := "train"]
  test_df[, split := "test"]
  
  train_df[, split_row := .I]
  test_df[, split_row := .I]
  
  
  all_df <- rbindlist(
    list(
      train_df,
      test_df
    ),
    use.names = TRUE,
    fill = TRUE
  )
  
  
  # ==========================================================
  # PYTHON GESTITO DA R
  # ==========================================================
  
  cat("\nPreparo Python e UmBERTo...\n")
  
  
  # reticulate crea automaticamente un ambiente Python
  py_require(
    c(
      "numpy",
      "torch",
      "transformers",
      "safetensors"
    )
  )
  
  
  # ==========================================================
  # CODICE PYTHON PER GLI EMBEDDINGS
  # ==========================================================
  
  py_run_string(
    '
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)


model.eval()


def make_embeddings(texts, batch_size=8):

    texts = list(texts)

    output = []

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]


        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )


        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }


        with torch.no_grad():

            hidden = model(
                **encoded
            ).last_hidden_state


            mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .to(hidden.dtype)
            )


            # masked mean pooling
            pooled = (
                (hidden * mask).sum(dim=1)
                /
                mask.sum(dim=1).clamp_min(1)
            )


        output.append(
            pooled.cpu().numpy()
        )


        done = min(
            start + batch_size,
            len(texts)
        )


        if done % 100 == 0 or done == len(texts):

            print(
                f"Embedded {done}/{len(texts)}"
            )


    return np.concatenate(
        output,
        axis=0
    )
'
  )


# ==========================================================
# GENERA EMBEDDINGS
# ==========================================================

cat(
  "\nGenero gli embeddings dei 2703 documenti...\n"
)


embeddings <- py$make_embeddings(
  as.character(all_df$testo_512),
  8L
)


embeddings <- as.matrix(
  embeddings
)


cat(
  "\nEmbedding matrix:",
  nrow(embeddings),
  "x",
  ncol(embeddings),
  "\n"
)


if (ncol(embeddings) != 768) {
  stop("Gli embeddings non hanno dimensione 768.")
}


# ==========================================================
# SALVA EMBEDDINGS
# ==========================================================

design <- list(
  
  metadata = all_df[
    ,
    .(
      split,
      split_row,
      pedianet_id,
      localita,
      diagnosi
    )
  ],
  
  embeddings = embeddings
)


saveRDS(
  design,
  EMBEDDING_CACHE
)


cat(
  "\nEmbeddings salvati in:\n",
  EMBEDDING_CACHE,
  "\n"
)


} else {
  
  cat(
    "\nUso embeddings già esistenti:\n",
    EMBEDDING_CACHE,
    "\n"
  )
}


# ============================================================
# 5. CARICA EMBEDDINGS
# ============================================================

design <- readRDS(
  EMBEDDING_CACHE
)


metadata <- as.data.table(
  design$metadata
)


embeddings <- as.matrix(
  design$embeddings
)


train_idx <- which(
  metadata$split == "train"
)


test_idx <- which(
  metadata$split == "test"
)


train_meta <- metadata[
  train_idx
]


test_meta <- metadata[
  test_idx
]


X_train <- embeddings[
  train_idx,
  ,
  drop = FALSE
]


X_test <- embeddings[
  test_idx,
  ,
  drop = FALSE
]


cat(
  "\n========================================\n"
)

cat(
  "DATASET\n"
)

cat(
  "========================================\n"
)

cat(
  "Train:",
  nrow(train_meta),
  "\n"
)

cat(
  "Test:",
  nrow(test_meta),
  "\n"
)

cat(
  "Localities:",
  uniqueN(train_meta$localita),
  "\n"
)


# ============================================================
# 6. STANDARDIZZAZIONE
#    FIT SOLO SUL TRAIN
# ============================================================

train_mean <- colMeans(
  X_train
)


train_sd <- apply(
  X_train,
  2,
  sd
)


train_sd[
  is.na(train_sd)
  |
    train_sd == 0
] <- 1


X_train_scaled <- sweep(
  X_train,
  2,
  train_mean,
  "-"
)


X_train_scaled <- sweep(
  X_train_scaled,
  2,
  train_sd,
  "/"
)


X_test_scaled <- sweep(
  X_test,
  2,
  train_mean,
  "-"
)


X_test_scaled <- sweep(
  X_test_scaled,
  2,
  train_sd,
  "/"
)


# ============================================================
# 7. PCA
#    FIT SOLO SUL TRAIN
# ============================================================

cat(
  "\nCalcolo PCA...\n"
)


pca <- prcomp(
  X_train_scaled,
  center = FALSE,
  scale. = FALSE
)


variance_explained <- (
  pca$sdev^2
  /
    sum(pca$sdev^2)
)


cum_variance <- cumsum(
  variance_explained
)


cat(
  "\nVarianza spiegata dai primi 30 PC:",
  round(
    cum_variance[K],
    4
  ),
  "\n"
)


PC_train <- as.data.table(
  pca$x[
    ,
    1:K,
    drop = FALSE
  ]
)


PC_test <- as.data.table(
  predict(
    pca,
    newdata = X_test_scaled
  )[
    ,
    1:K,
    drop = FALSE
  ]
)


pc_names <- paste0(
  "PC",
  1:K
)


setnames(
  PC_train,
  pc_names
)


setnames(
  PC_test,
  pc_names
)


# ============================================================
# 8. COSTRUISCI DATASET GLMM
# ============================================================

train_model <- cbind(
  
  train_meta[
    ,
    .(
      pedianet_id,
      localita,
      diagnosi
    )
  ],
  
  PC_train
)


test_model <- cbind(
  
  test_meta[
    ,
    .(
      split_row,
      pedianet_id,
      localita,
      diagnosi
    )
  ],
  
  PC_test
)


# Target
train_model[
  ,
  diagnosi := factor(
    diagnosi,
    levels = CLASSES
  )
]


test_model[
  ,
  diagnosi := factor(
    diagnosi,
    levels = CLASSES
  )
]


# Locality
train_model[
  ,
  localita := factor(
    localita
  )
]


test_model[
  ,
  localita := factor(
    localita,
    levels = levels(
      train_model$localita
    )
  )
]


if (
  any(
    is.na(test_model$localita)
  )
) {
  
  stop(
    "Nel test esiste una località non presente nel train."
  )
}


# ============================================================
# 9. FORMULA GLMM
# ============================================================

formula_glmm <- as.formula(
  
  paste(
    "diagnosi ~",
    paste(
      pc_names,
      collapse = " + "
    )
  )
)


# ============================================================
# 10. FIT DEL GLMM
# ============================================================

cat(
  "\n========================================\n"
)

cat(
  "FIT MULTINOMIAL GLMM\n"
)

cat(
  "========================================\n"
)


fit <- mblogit(
  
  formula = formula_glmm,
  
  data = train_model,
  
  random = ~ 1 | localita,
  
  catCov = "diagonal",
  
  method = "PQL",
  
  estimator = "ML"
)


cat(
  "\nGLMM stimato.\n"
)


saveRDS(
  fit,
  MODEL_FILE
)


# ============================================================
# 11. PREDIZIONI SUL TEST
# ============================================================

prob <- predict(
  
  fit,
  
  newdata = test_model,
  
  type = "response",
  
  conditional = TRUE
)


prob <- as.matrix(
  prob
)


cat(
  "\nClassi restituite dal GLMM:\n"
)

print(
  colnames(prob)
)


if (
  !all(
    CLASSES %in% colnames(prob)
  )
) {
  
  stop(
    "Le classi restituite dal GLMM non corrispondono alle 4 diagnosi."
  )
}


# Ordine classi
prob <- prob[
  ,
  CLASSES,
  drop = FALSE
]


y_true <- as.character(
  test_model$diagnosi
)


# ============================================================
# 12. FUNZIONE AUC BINARIA
# ============================================================

auc_binary <- function(
    y,
    score
) {
  
  n_pos <- sum(
    y == 1
  )
  
  n_neg <- sum(
    y == 0
  )
  
  
  if (
    n_pos == 0
    ||
    n_neg == 0
  ) {
    
    return(
      NA_real_
    )
  }
  
  
  ranks <- rank(
    score,
    ties.method = "average"
  )
  
  
  auc <- (
    
    sum(
      ranks[
        y == 1
      ]
    )
    
    -
      
      n_pos
    *
      (
        n_pos + 1
      )
    /
      2
    
  ) / (
    
    n_pos
    *
      n_neg
    
  )
  
  
  return(
    auc
  )
}


# ============================================================
# 13. FUNZIONE METRICHE
# ============================================================

compute_metrics <- function(
    y,
    probability
) {
  
  
  # ----------------------------------------------------------
  # Predicted class
  # ----------------------------------------------------------
  
  pred <- CLASSES[
    max.col(
      probability,
      ties.method = "first"
    )
  ]
  
  
  # ----------------------------------------------------------
  # Confusion matrix
  # ----------------------------------------------------------
  
  cm <- table(
    
    factor(
      y,
      levels = CLASSES
    ),
    
    factor(
      pred,
      levels = CLASSES
    )
  )
  
  
  tp <- diag(
    cm
  )
  
  
  n_true <- rowSums(
    cm
  )
  
  
  n_pred <- colSums(
    cm
  )
  
  
  # ----------------------------------------------------------
  # Recall
  # ----------------------------------------------------------
  
  recall <- tp / n_true
  
  
  # ----------------------------------------------------------
  # Precision
  # ----------------------------------------------------------
  
  precision <- ifelse(
    
    n_pred > 0,
    
    tp / n_pred,
    
    0
  )
  
  
  # ----------------------------------------------------------
  # F1
  # ----------------------------------------------------------
  
  f1 <- ifelse(
    
    precision + recall > 0,
    
    2
    *
      precision
    *
      recall
    /
      (
        precision
        +
          recall
      ),
    
    0
  )
  
  
  # ----------------------------------------------------------
  # AUC one-vs-rest
  # ----------------------------------------------------------
  
  aucs <- sapply(
    
    CLASSES,
    
    function(cl) {
      
      auc_binary(
        
        as.integer(
          y == cl
        ),
        
        probability[
          ,
          cl
        ]
      )
    }
  )
  
  
  # ----------------------------------------------------------
  # One-hot target per Brier
  # ----------------------------------------------------------
  
  one_hot <- sapply(
    
    CLASSES,
    
    function(cl) {
      
      as.numeric(
        y == cl
      )
    }
  )
  
  
  # ----------------------------------------------------------
  # Metriche finali
  # ----------------------------------------------------------
  
  c(
    
    Accuracy = mean(
      pred == y
    ),
    
    `Balanced Accuracy` = mean(
      recall
    ),
    
    `Macro-F1` = mean(
      f1
    ),
    
    `Macro-AUC` = mean(
      aucs,
      na.rm = TRUE
    ),
    
    Brier = mean(
      (
        probability
        -
          one_hot
      )^2
    )
  )
}


# ============================================================
# 14. STIME PUNTUALI
# ============================================================

point_estimates <- compute_metrics(
  y_true,
  prob
)


cat(
  "\n========================================\n"
)

cat(
  "STIME PUNTUALI\n"
)

cat(
  "========================================\n"
)


print(
  round(
    point_estimates,
    4
  )
)


# ============================================================
# 15. PATIENT-CLUSTER BOOTSTRAP
# ============================================================

cat(
  "\n========================================\n"
)

cat(
  "BOOTSTRAP 95% CI\n"
)

cat(
  "========================================\n"
)


# Ogni paziente mantiene insieme tutti i suoi documenti
patient_rows <- split(
  
  seq_len(
    nrow(test_model)
  ),
  
  as.character(
    test_model$pedianet_id
  )
)


patients <- names(
  patient_rows
)


boot_results <- matrix(
  
  NA_real_,
  
  nrow = N_BOOT,
  
  ncol = length(
    point_estimates
  )
)


colnames(
  boot_results
) <- names(
  point_estimates
)


set.seed(
  SEED + 1000
)


for (
  b in seq_len(N_BOOT)
) {
  
  
  sampled_patients <- sample(
    
    patients,
    
    size = length(
      patients
    ),
    
    replace = TRUE
  )
  
  
  idx <- unlist(
    
    patient_rows[
      sampled_patients
    ],
    
    use.names = FALSE
  )
  
  
  boot_results[
    b,
  ] <- compute_metrics(
    
    y_true[
      idx
    ],
    
    prob[
      idx,
      ,
      drop = FALSE
    ]
  )
  
  
  if (
    b %% 250 == 0
  ) {
    
    cat(
      b,
      "/",
      N_BOOT,
      "\n"
    )
  }
}


# ============================================================
# 16. 95% CONFIDENCE INTERVALS
# ============================================================

CI_low <- apply(
  
  boot_results,
  
  2,
  
  quantile,
  
  probs = 0.025,
  
  na.rm = TRUE
)


CI_high <- apply(
  
  boot_results,
  
  2,
  
  quantile,
  
  probs = 0.975,
  
  na.rm = TRUE
)


# ============================================================
# 17. TABELLA FINALE
# ============================================================

results <- data.table(
  
  Metric = names(
    point_estimates
  ),
  
  Estimate = as.numeric(
    point_estimates
  ),
  
  CI_low = as.numeric(
    CI_low
  ),
  
  CI_high = as.numeric(
    CI_high
  )
)


cat(
  "\n========================================\n"
)

cat(
  "RISULTATI FINALI GLMM\n"
)

cat(
  "========================================\n"
)


print(
  results
)


fwrite(
  results,
  RESULT_FILE
)


# ============================================================
# 18. SALVA PREDIZIONI
# ============================================================

predicted_class <- CLASSES[
  max.col(
    prob,
    ties.method = "first"
  )
]


predictions <- data.table(
  
  split_row = test_model$split_row,
  
  pedianet_id = as.character(
    test_model$pedianet_id
  ),
  
  localita = as.character(
    test_model$localita
  ),
  
  y_true = y_true,
  
  prediction = predicted_class
)


for (
  cl in CLASSES
) {
  
  predictions[
    ,
    paste0(
      "prob_",
      cl
    ) := prob[
      ,
      cl
    ]
  ]
}


fwrite(
  predictions,
  PRED_FILE
)


# ============================================================
# 19. FINE
# ============================================================

cat(
  "\n========================================\n"
)

cat(
  "FATTO\n"
)

cat(
  "========================================\n"
)


cat(
  "\nRisultati salvati in:\n",
  RESULT_FILE,
  "\n",
  sep = ""
)


cat(
  "\nPredizioni salvate in:\n",
  PRED_FILE,
  "\n",
  sep = ""
)


cat(
  "\nModello salvato in:\n",
  MODEL_FILE,
  "\n",
  sep = ""
)