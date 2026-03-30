import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tutel import moe as tutel_moe
from lib.experts.base import Expert
from lib.experts.config import ExpertConfig, SquadConfig, TrainingConfig, FeatureConfig
from lib.experts.node import HMoENode
import logging

import lib.experts.logging.registry

logger = logging.getLogger("FeatureRoutedExpert")


class FeatureRoutedExpert(Expert):
    """
    Feature-routed hierarchical Mixture-of-Experts (MoE) model.

    This class builds a tree of HMoENode routers and experts, performs forward
    passes with routing, computes losses, and manages training behaviors such
    as noise injection and optimizer configuration.
    """

    def __init__(self, config: ExpertConfig):
        """
        Initializes the FeatureRoutedExpert.

        Args:
            config (ExpertConfig): Full configuration object containing model,
                training, and hierarchical squad definitions.
        """
        super().__init__(config.expert_id, config.model_dim)

        self.full_config = config
        self.config = config.training

        # Base entropy regularization coefficient
        self.base_entropy_lambda = 0.01

        # Build root node depending on whether a single root squad exists
        if len(config.squads) == 1 and 'root' in config.squads:
            self.root_node = HMoENode("root", config.squads['root'])
        else:
            root_cfg = SquadConfig(
                model_dim=config.model_dim,
                num_experts=4,
                seq_len=64,
                children=config.squads
            )
            self.root_node = HMoENode("", root_cfg)

        # Log hierarchical structure
        logger.info("=== Hierarchical MoE Tree Structure ===")
        self._log_tree(self.root_node, indent="  ")
        logger.info("=======================================")

        # Spatial dropout applied to pooled features during training
        self.spatial_dropout = nn.Dropout1d(p=0.0)

    def _log_tree(self, node: HMoENode, indent: str = ""):
        """
        Recursively logs the hierarchical structure of the MoE tree.

        Args:
            node (HMoENode): Current node in the hierarchy.
            indent (str): Indentation string for pretty printing.
        """
        node_name = node.node_path if node.node_path else "ROOT_ROUTER"
        feat_info = f" (+ {len(node.config.features)} local features)" if node.has_local else ""
        router_info = " [LEAF]" if node.is_leaf else " [NEURAL ROUTER]"

        logger.info(f"{indent}➔ {node_name}{router_info}{feat_info}")

        if not node.is_leaf:
            for child_node in node.children_nodes.values():
                self._log_tree(child_node, indent + "    ")

    def forward_and_loss(
        self,
        batch_payload: dict,
        compute_loss: bool = True,
        return_features: bool = False
    ):
        """
        Performs forward pass and optionally computes loss.

        Args:
            batch_payload (dict): Input batch containing features and labels.
            compute_loss (bool): Whether to compute loss values.
            return_features (bool): If True, also return pooled features.

        Returns:
            Tuple[Tensor, dict] or Tuple[Tensor, Tensor]:
                Predictions and either loss dictionary or pooled features.
        """
        temp_noise = None

        # Disable gate noise during evaluation if previously set
        if not self.training and hasattr(self, 'current_gate_noise'):
            temp_noise = self.current_gate_noise
            self._apply_noise(0.0)

        # Extract inputs and targets
        features = batch_payload['features']
        targets = batch_payload['labels']['reversal_signal'] if 'labels' in batch_payload else None

        # Forward pass through hierarchical MoE
        root_logit, pooled_features, total_aux_loss, total_task_loss, shannon_entropy, routing_stats = self.root_node(
            features, targets
        )

        # Apply spatial dropout during training
        if self.training:
            pooled_features = self.spatial_dropout(pooled_features.unsqueeze(2)).squeeze(2)

        # Convert logits to probabilities (multi-class)
        predictions = F.softmax(root_logit, dim=1)

        # Return features if requested
        if return_features:
            if not self.training and temp_noise is not None:
                self._apply_noise(temp_noise)
            return predictions, pooled_features

        loss_dict = {}

        # Compute loss if targets are available
        if compute_loss and targets is not None:
            raw_targets_flat = targets.view(-1)

            # Initialize class targets
            class_targets = torch.zeros_like(raw_targets_flat, dtype=torch.long)

            # Map float targets (-1, 0, 1) to integer class indices
            for raw_val, map_info in self.root_node.config.class_map.items():
                class_targets[raw_targets_flat == float(raw_val)] = map_info['idx']

            # Cross-entropy loss for multi-class classification
            main_loss = F.cross_entropy(root_logit, class_targets)

            # Total loss combines main, task-specific, and auxiliary losses
            total_loss = main_loss + total_task_loss + (self.config.aux_loss_coef * total_aux_loss)

            # Populate loss dictionary
            loss_dict['total_loss'] = total_loss
            loss_dict['raw_logits'] = root_logit.detach()
            loss_dict['entropy'] = shannon_entropy.detach()
            loss_dict.update(routing_stats)

        # Restore gate noise after evaluation
        if not self.training and temp_noise is not None:
            self._apply_noise(temp_noise)

        return predictions, loss_dict

    def update_gate_noise(self, current_epoch: int, total_epochs: int):
        """
        Updates gate noise and router temperature based on training progress.

        Args:
            current_epoch (int): Current training epoch.
            total_epochs (int): Total number of epochs.
        """
        # Initial noise level
        self.initial_gate_noise = 0.1

        # Linear decay of noise over epochs
        decay_factor = max(0.0, 1.0 - (current_epoch / total_epochs))
        self.current_gate_noise = self.initial_gate_noise * decay_factor

        # Apply updated noise
        self._apply_noise(self.current_gate_noise)

        # Temperature decay (slower than noise decay)
        temp_decay = max(0.0, 1.0 - (current_epoch / (total_epochs * 0.8)))
        new_temp = 0.15 + (1.05 * temp_decay)

        # Update temperature for all routing nodes
        for m in self.modules():
            if isinstance(m, HMoENode):
                m.router_temperature = new_temp

    def _apply_noise(self, noise_val):
        """
        Applies gating noise to all MoE layers.

        Args:
            noise_val (float): Noise value to apply to gating mechanism.
        """

        def _set(module):
            # Set gate noise if module contains a Tutel MoE gate
            if hasattr(module, '_moe_layer') and hasattr(module._moe_layer, 'gate'):
                module._moe_layer.gate.gate_noise = noise_val

        # Traverse modules and apply noise to MoE layers
        for m in self.modules():
            if isinstance(m, tutel_moe.moe_layer):
                _set(m)

    def configure_optimizer(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer.
        """
        return optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
