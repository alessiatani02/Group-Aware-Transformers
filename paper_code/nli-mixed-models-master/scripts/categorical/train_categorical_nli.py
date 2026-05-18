import argparse, json, torch
import numpy as np
import os
import pandas as pd
from datetime import datetime

from nli_mixed_models.trainers.nli_trainer import (
    CategoricalRandomInterceptsTrainer,
    CategoricalRandomSlopesTrainer,
)
from nli_mixed_models.eval.nli_eval import CategoricalEval
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_veridicality,
    generate_splits,
    save_model_with_args,
)

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

LOG = setup_logging()

def compute_multiclass_metrics(y_true, y_pred, y_prob):
    labels = np.sort(np.unique(y_true))

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        metrics["auc_ovr_macro"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        metrics["auc_ovr_macro"] = np.nan

    specificities = []

    for label in labels:
        y_true_bin = (y_true == label).astype(int)
        y_pred_bin = (y_pred == label).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin,
            y_pred_bin,
            labels=[0, 1]
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        specificities.append(specificity)

    metrics["specificity_macro"] = np.nanmean(specificities)

    return metrics


def main(args):

    print("=== JOB STARTED ===", flush=True)
    # Load parameters from the supplied parameters file
    with open(args.parameters) as f:
        params = json.load(f)

    settings = params["settings"]
    checkpoints = params["checkpoints"]
    save_checkpoints = checkpoints["save_ckpts"]
    checkpoint_dir = checkpoints["ckpt_dir"]
    checkpoint_file_name = checkpoints["ckpt_file_name"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    results_dir = "/work/u10767335/results"
    os.makedirs(results_dir, exist_ok=True)

    for hyperparams in parameter_grid(params["hyper"]):
        for trainparams in parameter_grid(params["training"]):

            print("START LOADING CSV", flush=True)
            LOG.info("Loading Amazon dataset...")
            ver = pd.read_csv("/work/u10767335/data/amazon_model_dataset_full_encoded.csv")
            #per il test
            # ---- SAMPLE SUI GRUPPI ----
            target_n_samples = 50000

            products = ver["participant"].unique()
            np.random.shuffle(products)

            selected_products = []
            current_n = 0

            for p in products:
                group_size = (ver["participant"] == p).sum()
        
                selected_products.append(p)
                current_n += group_size
        
                if current_n >= target_n_samples:
                    break

            ver = ver[ver["participant"].isin(selected_products)]
            ver["participant"] = pd.factorize(ver["participant"])[0]

            print("N samples dopo group sampling:", len(ver))
            print("N groups:", ver["participant"].nunique())
            
            n_groups = ver["participant"].nunique()
            print(f"Numero di gruppi (products): {n_groups}")
            LOG.info(f"Number of groups (products): {n_groups}")

            ver["modal_response"] = ver["target"]
            ver["modal_response"] = ver["modal_response"].astype(int)
            ver["target"] = ver["target"].astype(int)
            hyperparams["n_participants"] = ver.participant.nunique()
            hyperparams["n_items"] = len(ver)
            print("CSV LOADED", flush=True)
            LOG.info("...Complete.")

            n_samples = len(ver)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Log the device being used
            if hyperparams["device"] == "cuda" and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            LOG.info(f"Using device {device}")

            # Assign folds to each veridicality example
            k_folds = settings["k_folds"]
            split_type = settings["split_type"]
            setting = hyperparams["setting"]
            ver = generate_splits(ver, split_type, k_folds=k_folds, datatype="v")
            ver = ver.reset_index(drop=True)
            LOG.info(
                f"Beginning training with {k_folds}-fold cross-validation in the {setting} setting, partitioning based on {split_type}."
            )
            
            # Save the DataFrame used for training + eval, in case
            # we need to recover it later.
            data_file_name = (
                checkpoint_dir
                + "/"
                + "-".join([checkpoint_file_name, "partition", split_type, "data"])
                + ".csv"
            )
            ver.to_csv(data_file_name, index=False)

            # Perform k-fold cross-validation
            # Note: we don't really need to be keeping track of the
            # random loss, since it should be constant. This is just
            # a sanity check.
            loss_all = []
            fixed_loss_all = []
            random_loss_all = []
            acc_all = []
            best_all = []

            loss_all_b = []
            fixed_loss_all_b = []
            random_loss_all_b = []
            acc_all_b = []
            best_all_b = []

            fold_results = []
            all_epochs_results = [] 
            all_random_effects = [] 

            for i in range(k_folds):

                # Dump current training settings
                LOG.info(
                    "Initializing categorical NLI model with the following hyperparameters:"
                )
                LOG.info(json.dumps(hyperparams, indent=4))
                LOG.info("Using the following training parameters:")
                LOG.info(json.dumps(trainparams, indent=4))
                LOG.info("And the following settings:")
                LOG.info(json.dumps(settings, indent=4))

                # Only the random intercepts models have been modified to handle
                # the standard setting in addition to the extended one.
                if (
                    hyperparams["setting"] == "extended"
                    and settings["use_random_slopes"]
                ):
                    cnli_trainer = CategoricalRandomSlopesTrainer(**hyperparams)
                else:
                    cnli_trainer = CategoricalRandomInterceptsTrainer(**hyperparams)

                # Select the folds
                test_fold = i
                train_folds = [j for j in range(k_folds) if j != test_fold]
                LOG.info(f"Beginning training with test fold={test_fold}.")

                train_data = ver[ver.fold.isin(train_folds)]
                test_data = ver[ver.fold == test_fold]

                # Fit the model on the train folds
                cat_model, history = cnli_trainer.fit(train_data=train_data, **trainparams)
                LOG.info("Finished training.")
                # === AGGIUNTO: salvataggio history per epoca ===
                epoch_df = pd.DataFrame({
                    "epoch": list(range(len(history["loss"]))),
                    "loss": history["loss"],
                    "fixed_loss": history["fixed_loss"],
                    "random_loss": history["random_loss"],
                    "accuracy": history["accuracy"],
                    "fold": i
                })

                all_epochs_results.append(epoch_df) 
                


                re = cat_model.random_effects.detach().cpu().numpy()
                print("\n=== RANDOM EFFECTS ===")
                print("Shape:", re.shape)
                print("Prime 10 righe:\n", re[:10])
                all_random_effects.append(re)

                # Save the model
                if save_checkpoints:
                    LOG.info("Saving model...")
                    model_name = (
                        "-".join(
                            [
                                checkpoint_file_name,
                                "fold",
                                str(i),
                                "partition",
                                split_type,
                            ]
                        )
                        + ".pt"
                    )
                    save_model_with_args(
                        params, cat_model, hyperparams, checkpoint_dir, model_name
                    )
                    LOG.info("Model saved.")

                # Evaluate the model on the test fold on subtask 'a'
                LOG.info("Evaluating subtask a")
                cnli_eval = CategoricalEval(cat_model, "a", device=device)
                (
                    loss_mean,
                    fixed_loss_mean,
                    random_loss_mean,
                    acc_mean,
                    best_mean,
                    spearman_mean,
                    best_spearman_mean,
                    _,
                    extra_outputs_a
                ) = cnli_eval.eval(test_data, trainparams["batch_size"])
                loss_all.append(loss_mean)
                fixed_loss_all.append(fixed_loss_mean)
                random_loss_all.append(random_loss_mean)
                acc_all.append(acc_mean)
                best_all.append(best_mean)

                metrics_a = compute_multiclass_metrics(
                    extra_outputs_a["y_true"],
                    extra_outputs_a["y_pred"],
                    extra_outputs_a["y_prob"]
                )


                LOG.info(f"Test results for fold {i}, subtask a")
                LOG.info(f"Mean loss:           {loss_mean}")
                LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
                LOG.info(f"Mean random loss:    {random_loss_mean}")
                LOG.info(f"Mean accuracy:       {acc_mean}")
                LOG.info(f"Prop. best possible: {best_mean}")
                LOG.info(f"Balanced accuracy:   {metrics_a['balanced_accuracy']}")
                LOG.info(f"Sensitivity macro:   {metrics_a['sensitivity_macro']}")
                LOG.info(f"Specificity macro:   {metrics_a['specificity_macro']}")
                LOG.info(f"AUC OVR macro:       {metrics_a['auc_ovr_macro']}")

                # Evaluate the model on the test fold on subtask 'b'
                LOG.info("Evaluating subtask b")
                cnli_eval = CategoricalEval(cat_model, "b", device=device)
                (
                    loss_mean,
                    fixed_loss_mean,
                    random_loss_mean,
                    acc_mean,
                    best_mean,
                    spearman_mean,
                    best_spearman_mean,
                    _,
                    extra_outputs_b
                ) = cnli_eval.eval(test_data, trainparams["batch_size"])
                loss_all_b.append(loss_mean)
                fixed_loss_all_b.append(fixed_loss_mean)
                random_loss_all_b.append(random_loss_mean)
                acc_all_b.append(acc_mean)
                best_all_b.append(best_mean)

                metrics_b = compute_multiclass_metrics(
                    extra_outputs_b["y_true"],
                    extra_outputs_b["y_pred"],
                    extra_outputs_b["y_prob"]
                )


                LOG.info(f"Test results for fold {i}, subtask b")
                LOG.info(f"Mean loss:           {loss_mean}")
                LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
                LOG.info(f"Mean random loss:    {random_loss_mean}")
                LOG.info(f"Mean accuracy:       {acc_mean}")
                LOG.info(f"Prop. best possible: {best_mean}")
                LOG.info(f"Balanced accuracy:   {metrics_b['balanced_accuracy']}")
                LOG.info(f"Sensitivity macro:   {metrics_b['sensitivity_macro']}")
                LOG.info(f"Specificity macro:   {metrics_b['specificity_macro']}")
                LOG.info(f"AUC OVR macro:       {metrics_b['auc_ovr_macro']}")
                # === AGGIUNTO: salvataggio risultati per fold ===
                fold_results.append({
                    "fold": i,
                    "loss_a": loss_all[-1],
                    "fixed_loss_a": fixed_loss_all[-1],
                    "random_loss_a": random_loss_all[-1],
                    "acc_a": acc_all[-1],
                    "loss_b": loss_all_b[-1],
                    "fixed_loss_b": fixed_loss_all_b[-1],
                    "random_loss_b": random_loss_all_b[-1],
                    "acc_b": acc_all_b[-1],
                    "balanced_accuracy_a": metrics_a["balanced_accuracy"],
                    "sensitivity_macro_a": metrics_a["sensitivity_macro"],
                    "specificity_macro_a": metrics_a["specificity_macro"],
                    "precision_macro_a": metrics_a["precision_macro"],
                    "f1_macro_a": metrics_a["f1_macro"],
                    "auc_ovr_macro_a": metrics_a["auc_ovr_macro"],

                    "balanced_accuracy_b": metrics_b["balanced_accuracy"],
                    "sensitivity_macro_b": metrics_b["sensitivity_macro"],
                    "specificity_macro_b": metrics_b["specificity_macro"],
                    "precision_macro_b": metrics_b["precision_macro"],
                    "f1_macro_b": metrics_b["f1_macro"],
                    "auc_ovr_macro_b": metrics_b["auc_ovr_macro"],
                })

            df_folds = pd.DataFrame(fold_results)
            LOG.info("Finished k-fold cross evaluation.")
            LOG.info("Subtask a results:")
            LOG.info(f"Mean loss:           {np.round(np.mean(loss_all), 4)} ± {np.round(np.std(loss_all), 4)}")
            LOG.info(f"Mean fixed loss:     {np.round(np.mean(fixed_loss_all), 4)} ± {np.round(np.std(fixed_loss_all), 4)}")
            LOG.info(f"Mean random loss:    {np.round(np.mean(random_loss_all), 4)} ± {np.round(np.std(random_loss_all), 4)}")
            LOG.info(f"Mean accuracy:       {np.round(np.mean(acc_all), 4)} ± {np.round(np.std(acc_all), 4)}")
            LOG.info(f"Prop. best possible: {np.round(np.mean(best_all), 4)} ± {np.round(np.std(best_all), 4)}")
            LOG.info(f"Mean balanced accuracy: {np.round(df_folds['balanced_accuracy_a'].mean(), 4)} ± {np.round(df_folds['balanced_accuracy_a'].std(), 4)}")
            LOG.info(f"Mean sensitivity macro: {np.round(df_folds['sensitivity_macro_a'].mean(), 4)} ± {np.round(df_folds['sensitivity_macro_a'].std(), 4)}")
            LOG.info(f"Mean specificity macro: {np.round(df_folds['specificity_macro_a'].mean(), 4)} ± {np.round(df_folds['specificity_macro_a'].std(), 4)}")
            LOG.info(f"Mean AUC OVR macro:     {np.round(df_folds['auc_ovr_macro_a'].mean(), 4)} ± {np.round(df_folds['auc_ovr_macro_a'].std(), 4)}")
            LOG.info(f"Mean F1 macro:          {np.round(df_folds['f1_macro_a'].mean(), 4)} ± {np.round(df_folds['f1_macro_a'].std(), 4)}")
            
            LOG.info("Subtask b results:")
            LOG.info(f"Mean loss:           {np.round(np.mean(loss_all_b), 4)} ± {np.round(np.std(loss_all_b), 4)}")
            LOG.info(f"Mean fixed loss:     {np.round(np.mean(fixed_loss_all_b), 4)} ± {np.round(np.std(fixed_loss_all_b), 4)}")
            LOG.info(f"Mean random loss:    {np.round(np.mean(random_loss_all_b), 4)} ± {np.round(np.std(random_loss_all_b), 4)}")
            LOG.info(f"Mean accuracy:       {np.round(np.mean(acc_all_b), 4)} ± {np.round(np.std(acc_all_b), 4)}")
            LOG.info(f"Prop. best possible: {np.round(np.mean(best_all_b), 4)} ± {np.round(np.std(best_all_b), 4)}")
            LOG.info(f"Mean balanced accuracy: {np.round(df_folds['balanced_accuracy_b'].mean(), 4)} ± {np.round(df_folds['balanced_accuracy_b'].std(), 4)}")
            LOG.info(f"Mean sensitivity macro: {np.round(df_folds['sensitivity_macro_b'].mean(), 4)} ± {np.round(df_folds['sensitivity_macro_b'].std(), 4)}")
            LOG.info(f"Mean specificity macro: {np.round(df_folds['specificity_macro_b'].mean(), 4)} ± {np.round(df_folds['specificity_macro_b'].std(), 4)}")
            LOG.info(f"Mean AUC OVR macro:     {np.round(df_folds['auc_ovr_macro_b'].mean(), 4)} ± {np.round(df_folds['auc_ovr_macro_b'].std(), 4)}")
            LOG.info(f"Mean F1 macro:          {np.round(df_folds['f1_macro_b'].mean(), 4)} ± {np.round(df_folds['f1_macro_b'].std(), 4)}")
            # === AGGIUNTO: salvataggio CSV ===

            df_epochs = pd.concat(all_epochs_results, ignore_index=True)

            

            df_epochs.to_csv(
                f"{results_dir}/new_epochs_ALL_{n_samples}_{timestamp}.csv",
                index=False
            )

            # --- risultati per fold ---
            df_folds.to_csv(
                f"{results_dir}/new_results_{n_samples}_{timestamp}.csv",
                index=False
            )

            # --- summary finale ---
            df_summary = pd.DataFrame([{
                "n_samples": n_samples,
                 "n_groups": n_groups,
                 "lambda_random": trainparams["lambda_random"],

                "mean_loss_a": np.mean(loss_all),
                "std_loss_a": np.std(loss_all),

                "mean_fixed_loss_a": np.mean(fixed_loss_all),
                "std_fixed_loss_a": np.std(fixed_loss_all),

                "mean_random_loss_a": np.mean(random_loss_all),
                "std_random_loss_a": np.std(random_loss_all),

                "mean_acc_a": np.mean(acc_all),
                "std_acc_a": np.std(acc_all),

                "mean_loss_b": np.mean(loss_all_b),
                "std_loss_b": np.std(loss_all_b),

                "mean_fixed_loss_b": np.mean(fixed_loss_all_b),
                "std_fixed_loss_b": np.std(fixed_loss_all_b),

                "mean_random_loss_b": np.mean(random_loss_all_b),
                "std_random_loss_b": np.std(random_loss_all_b),

                "mean_acc_b": np.mean(acc_all_b),
                "std_acc_b": np.std(acc_all_b),

                "mean_balanced_accuracy_a": df_folds["balanced_accuracy_a"].mean(),
                "std_balanced_accuracy_a": df_folds["balanced_accuracy_a"].std(),

                "mean_sensitivity_macro_a": df_folds["sensitivity_macro_a"].mean(),
                "std_sensitivity_macro_a": df_folds["sensitivity_macro_a"].std(),

                "mean_specificity_macro_a": df_folds["specificity_macro_a"].mean(),
                "std_specificity_macro_a": df_folds["specificity_macro_a"].std(),

                "mean_precision_macro_a": df_folds["precision_macro_a"].mean(),
                "std_precision_macro_a": df_folds["precision_macro_a"].std(),

                "mean_f1_macro_a": df_folds["f1_macro_a"].mean(),
                "std_f1_macro_a": df_folds["f1_macro_a"].std(),

                "mean_auc_ovr_macro_a": df_folds["auc_ovr_macro_a"].mean(),
                "std_auc_ovr_macro_a": df_folds["auc_ovr_macro_a"].std(),

                "mean_balanced_accuracy_b": df_folds["balanced_accuracy_b"].mean(),
                "std_balanced_accuracy_b": df_folds["balanced_accuracy_b"].std(),

                "mean_sensitivity_macro_b": df_folds["sensitivity_macro_b"].mean(),
                "std_sensitivity_macro_b": df_folds["sensitivity_macro_b"].std(),

                "mean_specificity_macro_b": df_folds["specificity_macro_b"].mean(),
                "std_specificity_macro_b": df_folds["specificity_macro_b"].std(),

                "mean_precision_macro_b": df_folds["precision_macro_b"].mean(),
                "std_precision_macro_b": df_folds["precision_macro_b"].std(),

                "mean_f1_macro_b": df_folds["f1_macro_b"].mean(),
                "std_f1_macro_b": df_folds["f1_macro_b"].std(),

                "mean_auc_ovr_macro_b": df_folds["auc_ovr_macro_b"].mean(),
                "std_auc_ovr_macro_b": df_folds["auc_ovr_macro_b"].std(),
            }])

            df_summary.to_csv(
                f"{results_dir}/new_summary_{n_samples}_{timestamp}.csv",
                index=False
            )


             #=== AGGIUNTO: salvataggio unico random effects ===
            if len(all_random_effects) > 0:
                re_all = np.concatenate(all_random_effects, axis=0)

                np.save(
                    f"{results_dir}/new_random_effects_ALL_{n_samples}_{timestamp}.npy",
                    re_all
                )




            print("=== TRAINING FINISHED ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a categorical NLI mixed effects model"
    )
    parser.add_argument(
        "--parameters",
        type=str,
        default="scripts/categorical/train_categorical_nli.json",
        help="Path to a JSON containing parameters to sweep",
    )
    args = parser.parse_args()
    main(args)
