import logging
import sys
import torch

from lib.experts.routed import FeatureRoutedExpert
from lib.experts.controller.unified import UnifiedTrainingController
from lib.experts.data.pipeline import MoEDataPipeline

import config


import lib.experts.logging.registry

logger = logging.getLogger("Train")


def _set_input_dims_recursively(squads_dict, feature_map, prefix=""):
    """
    Recursively traverses the hierarchical Squad configuration to dynamically inject 
    the correct input dimensions based on the actual fetched data.

    This function acts as a critical bridge between the data pipeline and the model 
    architecture. Because features are dynamically requested in `config.py` (and some 
    may fail to fetch due to API errors or NaNs), the network cannot hardcode its 
    input dimensions. This recursive walk ensures every sub-router and leaf node 
    is initialized with the exact number of features it will actually receive.

    Args:
        squads_dict (dict): The nested dictionary of SquadConfig objects.
        feature_map (dict): A flat dictionary mapping node paths to their successfully fetched feature lists.
        prefix (str, optional): The accumulated dot-notation path of the current node. Defaults to "".
    """
    for k, v in squads_dict.items():
        node_path = f"{prefix}.{k}" if prefix else k
        if node_path in feature_map:
            v.input_dim = len(feature_map[node_path])
        if v.children:
            _set_input_dims_recursively(v.children, feature_map, node_path)

def main():
    """
    The primary execution entry point for training the Hierarchical Mixture of Experts (HMoE).

    This function orchestrates the entire lifecycle:
    1. Validates hardware and configuration parameters.
    2. Triggers the MoEDataPipeline to fetch, normalize, and window historical market data.
    3. Executes an aggressive memory optimization strategy (VRAM pre-loading).
    4. Dynamically sizes and compiles the HMoE architecture.
    5. Hands off execution to the UnifiedTrainingController.
    """
    SYMBOL = config.SYMBOL
    TIMEFRAME = config.TIMEFRAME
    DEVICE = config.DEVICE

    # Enforce strict configuration requirements to prevent downstream crashes during pipeline execution.
    if not hasattr(config, 'SQUADS'):
        logger.error("CRITICAL: 'SQUADS' dictionary not found in config.py.")
        sys.exit(1)
        
    SQUADS_CONFIG = config.SQUADS

    # Microsoft Tutel's MoE primitives rely heavily on custom CUDA kernels for efficient 
    # all-to-all communication and sparse matrix multiplication.
    if DEVICE.type != 'cuda':
        logger.warning("Tutel is optimized for CUDA. Training on CPU will be slow.")

    pipeline = MoEDataPipeline()

    logger.info(f"Initiating dynamic routing pipeline for {SYMBOL} on {TIMEFRAME}...")

    # Phase 1: Data Acquisition
    # Fetch the training data and retrieve the specific noise profiles configured for data augmentation.
    squad_tensors_train, Y_train, squad_noise_maps_train = pipeline.fetch_squads_and_window(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=config.START_DATE_TRAIN,
        end_date=config.END_DATE_TRAIN,
        squads_config=SQUADS_CONFIG,
        target_feature=config.TARGET_FEATURE,
    )

    # Fetch the validation data. Note that noise_maps are intentionally discarded (_) here, 
    # as data augmentation (Gaussian mutilation) should never be applied during validation.
    squad_tensors_val, Y_val, _ = pipeline.fetch_squads_and_window(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=config.START_DATE_VAL,
        end_date=config.END_DATE_VAL,
        squads_config=SQUADS_CONFIG,
        target_feature=config.TARGET_FEATURE,
    )

    if Y_train.nelement() == 0 or Y_val.nelement() == 0:
        logger.error("Failed to fetch enough data to build tensors. Exiting.")
        sys.exit(1)

    # Phase 2: Memory Optimization Strategy
    # In standard PyTorch, data sits in RAM and is moved to VRAM batch-by-batch via the DataLoader, 
    # causing a massive PCIe bandwidth bottleneck. Because tabular financial data is relatively small 
    # (compared to images/video), we bypass this entirely by pre-loading the entire dataset directly 
    # onto the GPU before the training loop even begins.
    logger.info("Pre-loading entire dataset directly into VRAM to eliminate CPU bottlenecks...")

    for node_path in squad_tensors_train.keys():
        squad_tensors_train[node_path] = squad_tensors_train[node_path].to(DEVICE)
        
    if squad_noise_maps_train:
        for node_path in squad_noise_maps_train.keys():
            squad_noise_maps_train[node_path] = squad_noise_maps_train[node_path].to(DEVICE)
        
    Y_train = Y_train.to(DEVICE)

    # Phase 3: Fast DataLoader Instantiation
    # Initialize the custom FastDictDataLoaders. These loaders expect data to already be in VRAM 
    # and simply yield randomized index slices, drastically accelerating epoch times.
    train_loader = pipeline.build_dataloader(
        squad_tensors=squad_tensors_train, 
        Y=Y_train, 
        batch_size=64, 
        is_training=True,
        noise_maps=squad_noise_maps_train
    )

    val_loader = pipeline.build_dataloader(
        squad_tensors=squad_tensors_val, 
        Y=Y_val, 
        batch_size=64, 
        is_training=False
    )

    logger.info(f"Starting HMoE Training. Discovered Nodes -> [{', '.join(pipeline.last_squad_map.keys())}]")

    # Phase 4: Model Architecture Initialization
    expert_config = config.get_expert_config()
    
    # Synchronize the config with the actual dimensions of the successfully fetched data.
    _set_input_dims_recursively(expert_config.squads, pipeline.last_squad_map)

    expert = FeatureRoutedExpert(config=expert_config)

    # Phase 5: Kernel Fusion (PyTorch 2.0+)
    # torch.compile uses TorchDynamo to analyze the execution graph and fuse multiple 
    # operations into single custom Triton kernels. This reduces CUDA overhead and memory 
    # read/writes, significantly speeding up the complex MoE routing pathways.
    if hasattr(torch, 'compile'):
        logger.info("Fusing GPU kernels with torch.compile() to bypass CPU memory bottlenecks...")
        expert = torch.compile(expert)
    else:
        logger.warning("torch.compile() not available. Upgrade to PyTorch 2.0+ for massive speed boosts.")

    # Phase 6: Training Orchestration
    # Hand the compiled model, VRAM-resident loaders, and feature metadata over to the Unified Controller.
    controller = UnifiedTrainingController(
        expert=expert,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        checkpoint_dir='./checkpoints/hierarchical_moe',
        feature_names=pipeline.last_squad_map
    )

    controller.run_full_training(
        max_epochs=getattr(config, 'EPOCHS', 50),
        patience=getattr(config, 'PATIENCE', 15) 
    )

if __name__ == "__main__":
    main()