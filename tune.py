import logging
import sys
import os
import torch
import optuna

from lib.experts.routed import FeatureRoutedExpert
from lib.experts.controller.unified import UnifiedTrainingController
from lib.experts.data.pipeline import MoEDataPipeline

import config
import lib.experts.logging.registry

logger = logging.getLogger("Tuner")

def _set_input_dims_recursively(squads_dict, feature_map, prefix=""):
    """Helper to route dynamic input dimensions to the tree."""
    for k, v in squads_dict.items():
        node_path = f"{prefix}.{k}" if prefix else k
        if node_path in feature_map:
            v.input_dim = len(feature_map[node_path])
        if v.children:
            _set_input_dims_recursively(v.children, feature_map, node_path)

def objective(trial, train_loader, val_loader, feature_names):
    """The Optuna trial function."""
    DEVICE = config.DEVICE

    # 1. Define the strategic Multi-Class hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True) # The Noise Filter
    aux_loss_coef = trial.suggest_float("aux_loss_coef", 0.05, 0.5)          # The Expert Balancer
    pos_weight = trial.suggest_float("pos_weight", 0.8, 2.0)                 # The Recall Anchor

    # 2. Inject suggested hyperparams into the config
    expert_config = config.get_expert_config()
    _set_input_dims_recursively(expert_config.squads, feature_names)
    
    expert_config.training.lr = lr
    expert_config.training.weight_decay = weight_decay
    expert_config.training.aux_loss_coef = aux_loss_coef
    expert_config.training.pos_weight = pos_weight

    # 3. Initialize Model
    expert = FeatureRoutedExpert(config=expert_config)

    if hasattr(torch, 'compile'):
        expert = torch.compile(expert)

    # 4. Initialize Controller (Isolate checkpoints per trial so they don't overwrite)
    trial_ckpt_dir = f'./checkpoints/tuning_moe/trial_{trial.number}'
    os.makedirs(trial_ckpt_dir, exist_ok=True)

    controller = UnifiedTrainingController(
        expert=expert,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        checkpoint_dir=trial_ckpt_dir,
        feature_names=feature_names
    )

    max_epochs = getattr(config, 'EPOCHS', 50)
    max_epochs = 16
    best_f1 = 0.0

    # 5. Training Loop with Pruning
    for epoch in range(1, max_epochs + 1):
        if hasattr(expert, 'update_gate_noise'):
            expert.update_gate_noise(epoch, max_epochs)

        train_loss = controller.train_epoch(epoch)
        val_loss, val_f1 = controller.validate(epoch)

        best_f1 = max(best_f1, val_f1)

        # Report intermediate values to Optuna
        trial.report(val_f1, epoch)

        # Handle pruning (stop early if the trial is unpromising)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()

    return best_f1


def main():
    SYMBOL = config.SYMBOL
    TIMEFRAME = config.TIMEFRAME
    DEVICE = config.DEVICE
    SQUADS_CONFIG = config.SQUADS

    if DEVICE.type != 'cuda':
        logger.warning("Tutel is optimized for CUDA. Tuning on CPU will be extremely slow.")

    pipeline = MoEDataPipeline()
    logger.info(f"Initiating Optuna tuning pipeline for {SYMBOL} on {TIMEFRAME}...")

    # --- 1. GLOBAL DATA LOADING (Done only once!) ---
    squad_tensors_train, Y_train, squad_noise_maps_train = pipeline.fetch_squads_and_window(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=config.START_DATE_TRAIN,
        end_date=config.END_DATE_TRAIN,
        squads_config=SQUADS_CONFIG,
        target_feature=config.TARGET_FEATURE,
    )

    squad_tensors_val, Y_val, _ = pipeline.fetch_squads_and_window(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=config.START_DATE_VAL,
        end_date=config.END_DATE_VAL,
        squads_config=SQUADS_CONFIG,
        target_feature=config.TARGET_FEATURE,
    )

    if Y_train.nelement() == 0 or Y_val.nelement() == 0:
        logger.error("Failed to fetch enough data. Exiting.")
        sys.exit(1)

    logger.info("Pre-loading tuning dataset directly into VRAM...")

    for node_path in squad_tensors_train.keys():
        squad_tensors_train[node_path] = squad_tensors_train[node_path].to(DEVICE)
    if squad_noise_maps_train:
        for node_path in squad_noise_maps_train.keys():
            squad_noise_maps_train[node_path] = squad_noise_maps_train[node_path].to(DEVICE)
    Y_train = Y_train.to(DEVICE)

    train_loader = pipeline.build_dataloader(
        squad_tensors=squad_tensors_train, Y=Y_train, batch_size=64, 
        is_training=True, noise_maps=squad_noise_maps_train
    )

    val_loader = pipeline.build_dataloader(
        squad_tensors=squad_tensors_val, Y=Y_val, batch_size=64, is_training=False
    )

    feature_names = pipeline.last_squad_map

    # --- 2. OPTUNA SETUP ---
    study_name = f"moe_tune_{SYMBOL}_{TIMEFRAME}"
    storage_url = "sqlite:///optuna.db"

    # Use MedianPruner to kill unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    # Create the study and point it to the local SQLite database
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner
    )

    logger.info(f"Starting Optuna Study: {study_name}. Results saving to {storage_url}")

    # Pass the pre-loaded dataloaders to the objective function
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, feature_names),
        n_trials=50,
        n_jobs=1  # Keep at 1 for PyTorch CUDA stability unless using DDP
    )

    logger.info("=== OPTUNA TUNING COMPLETE ===")
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best F1 Score: {study.best_value:.4f}")
    logger.info("Best Parameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()