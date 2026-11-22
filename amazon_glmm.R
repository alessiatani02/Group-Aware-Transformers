
library(data.table)

# File del GLMM corretto
out_dir <- "glmm_results_corrected_trainPCA_stratified"

predictions <- fread(
  file.path(
    out_dir,
    "cellphones_GLMM_CORRECTED_K30_test_predictions.csv"
  )
)

y <- as.integer(predictions$y_true)
prob <- as.numeric(predictions$probability)

# Soglia scelta nella precedente analisi
threshold <- 0.88

# Funzione per calcolare le metriche
compute_metrics <- function(y, prob, threshold) {
  
  pred <- as.integer(prob >= threshold)
  
  TP <- sum(y == 1 & pred == 1)
  TN <- sum(y == 0 & pred == 0)
  FP <- sum(y == 0 & pred == 1)
  FN <- sum(y == 1 & pred == 0)
  
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  
  c(
    accuracy = mean(pred == y),
    balanced_accuracy = (sensitivity + specificity) / 2,
    sensitivity = sensitivity,
    specificity = specificity
  )
}

# Valori osservati
observed <- compute_metrics(y, prob, threshold)

# Bootstrap stratificato per classe
set.seed(123)
B <- 2000

idx0 <- which(y == 0)
idx1 <- which(y == 1)

boot <- replicate(B, {
  idx <- c(
    sample(idx0, length(idx0), replace = TRUE),
    sample(idx1, length(idx1), replace = TRUE)
  )
  
  compute_metrics(y[idx], prob[idx], threshold)
})

# Intervalli percentile 95%
ci <- t(apply(
  boot, 1, quantile,
  probs = c(0.025, 0.975)
))

results <- data.table(
  metric = names(observed),
  estimate = as.numeric(observed),
  ci95_low = ci[, 1],
  ci95_high = ci[, 2]
)

cat("\n========== GLMM THRESHOLD 0.88 ==========\n")
print(results)

# Salva
fwrite(
  results,
  file.path(out_dir, "GLMM_threshold_0p88_bootstrap_CI.csv")
)
