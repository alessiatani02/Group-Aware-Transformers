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

LOG = setup_logging()


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
            ver = pd.read_csv("/work/u10767335/data/amazon_model_dataset_full.csv")
            #per il test
            ver = ver.iloc[:50000]
            n_groups = ver["participant"].nunique()
            print(f"Numero di gruppi (products): {n_groups}")
            LOG.info(f"Number of groups (products): {n_groups}")


            ver["participant"] = pd.factorize(ver["participant"])[0]
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
            device = hyperparams["device"]
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
                    _
                ) = cnli_eval.eval(test_data, trainparams["batch_size"])
                loss_all.append(loss_mean)
                fixed_loss_all.append(fixed_loss_mean)
                random_loss_all.append(random_loss_mean)
                acc_all.append(acc_mean)
                best_all.append(best_mean)
                LOG.info(f"Test results for fold {i}, subtask a")
                LOG.info(f"Mean loss:           {loss_mean}")
                LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
                LOG.info(f"Mean random loss:    {random_loss_mean}")
                LOG.info(f"Mean accuracy:       {acc_mean}")
                LOG.info(f"Prop. best possible: {best_mean}")

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
                    _
                ) = cnli_eval.eval(test_data, trainparams["batch_size"])
                loss_all_b.append(loss_mean)
                fixed_loss_all_b.append(fixed_loss_mean)
                random_loss_all_b.append(random_loss_mean)
                acc_all_b.append(acc_mean)
                best_all_b.append(best_mean)
                LOG.info(f"Test results for fold {i}, subtask b")
                LOG.info(f"Mean loss:           {loss_mean}")
                LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
                LOG.info(f"Mean random loss:    {random_loss_mean}")
                LOG.info(f"Mean accuracy:       {acc_mean}")
                LOG.info(f"Prop. best possible: {best_mean}")
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
                })

            LOG.info("Finished k-fold cross evaluation.")
            LOG.info("Subtask a results:")
            LOG.info(f"Mean loss:           {np.round(np.mean(loss_all), 4)} ± {np.round(np.std(loss_all), 4)}")
            LOG.info(f"Mean fixed loss:     {np.round(np.mean(fixed_loss_all), 4)} ± {np.round(np.std(fixed_loss_all), 4)}")
            LOG.info(f"Mean random loss:    {np.round(np.mean(random_loss_all), 4)} ± {np.round(np.std(random_loss_all), 4)}")
            LOG.info(f"Mean accuracy:       {np.round(np.mean(acc_all), 4)} ± {np.round(np.std(acc_all), 4)}")
            LOG.info(f"Prop. best possible: {np.round(np.mean(best_all), 4)} ± {np.round(np.std(best_all), 4)}")
            
            LOG.info("Subtask b results:")
            LOG.info(f"Mean loss:           {np.round(np.mean(loss_all_b), 4)} ± {np.round(np.std(loss_all_b), 4)}")
            LOG.info(f"Mean fixed loss:     {np.round(np.mean(fixed_loss_all_b), 4)} ± {np.round(np.std(fixed_loss_all_b), 4)}")
            LOG.info(f"Mean random loss:    {np.round(np.mean(random_loss_all_b), 4)} ± {np.round(np.std(random_loss_all_b), 4)}")
            LOG.info(f"Mean accuracy:       {np.round(np.mean(acc_all_b), 4)} ± {np.round(np.std(acc_all_b), 4)}")
            LOG.info(f"Prop. best possible: {np.round(np.mean(best_all_b), 4)} ± {np.round(np.std(best_all_b), 4)}")

            # === AGGIUNTO: salvataggio CSV ===

            df_epochs = pd.concat(all_epochs_results, ignore_index=True)

            # === AGGIUNTO: salvataggio unico random effects ===
            if len(all_random_effects) > 0:
                re_all = np.vstack(all_random_effects)

                np.save(
                    f"{results_dir}/random_effects_ALL_{n_samples}_{timestamp}.npy",
                    re_all
                )

            df_epochs.to_csv(
                f"{results_dir}/epochs_ALL_{n_samples}_{timestamp}.csv",
                index=False
            )

            # --- risultati per fold ---
            df_folds = pd.DataFrame(fold_results)
            df_folds.to_csv(
                f"{results_dir}/results_{n_samples}_{timestamp}.csv",
                index=False
            )

            # --- summary finale ---
            df_summary = pd.DataFrame([{
                "n_samples": n_samples,
                 "n_groups": n_groups,

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
            }])

            df_summary.to_csv(
                f"{results_dir}/summary_{n_samples}_{timestamp}.csv",
                index=False
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
