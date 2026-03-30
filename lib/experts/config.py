import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Expert(nn.Module, ABC):
    """Base class for all RPulsar Expert models.

    This abstract class defines a standardized interface for expert models so
    they can be trained and evaluated by a centralized controller without
    requiring model-specific handling.

    Subclasses are required to implement:
        - forward_and_loss
        - configure_optimizer
    """

    def __init__(self, expert_id: str, model_dim: int):
        """Initialize an Expert instance.

        Args:
            expert_id (str): Unique identifier for the expert instance.
            model_dim (int): Dimensionality of internal model representations.
        """
        super().__init__()  # Initialize nn.Module base class

        self.expert_id = expert_id  # Store unique identifier for tracking/logging
        self.model_dim = model_dim  # Store internal feature dimension

    @abstractmethod
    def forward_and_loss(
        self,
        batch_payload: dict,
        compute_loss=True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward pass and optionally compute loss values.

        This method encapsulates both inference and training logic so that
        external controllers can remain model-agnostic.

        Args:
            batch_payload (dict): Input dictionary containing:
                - features (Dict[str, torch.Tensor]): Model inputs
                - labels (Dict[str, torch.Tensor]): Ground truth targets
            compute_loss (bool, optional): Whether to compute loss values.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                predictions (torch.Tensor): Model output tensor
                loss_dict (Dict[str, torch.Tensor]): Dictionary of computed losses
        """
        pass  # Must be implemented by subclass

    @abstractmethod
    def configure_optimizer(self) -> torch.optim.Optimizer:
        """Create and return the optimizer for this expert.

        Each expert can define its own optimization strategy and
        hyperparameters.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """
        pass  # Must be implemented by subclass


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class FeatureConfig:
    """Configuration for a single market feature.

    Defines how an individual feature should be processed before being
    consumed by the model.

    Args:
        name (str): Name of the feature.
        normalize (bool, optional): Whether to normalize the feature.
            Defaults to True.
    """
    name: str
    normalize: bool = True  # Whether to normalize this feature during preprocessing
    gaussian_noise_std: float = 0.0  # Standard deviation of Gaussian noise to add for regularization



@dataclass
class SquadConfig:
    """Configuration for a squad of experts.

    A squad groups multiple experts and defines how inputs are processed,
    routed, and regularized within that group.

    Args:
        input_dim (int, optional): Input dimensionality. Defaults to 0.
        model_dim (int, optional): Internal representation size. Defaults to 64.
        num_experts (int, optional): Number of experts in the squad. Defaults to 4.
        seq_len (int, optional): Sequence length for temporal inputs. Defaults to 64.
        features (List[FeatureConfig], optional): List of feature configs.
        children (Optional[Dict[str, SquadConfig]], optional): Nested child squads.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        routing_mode (str, optional): Routing strategy ("neural" or "deterministic").
        routing_feature (str, optional): Feature used for deterministic routing.
        routing_map (dict, optional): Mapping for deterministic routing.
    """
    input_dim: int = 0  # Input dimensionality for the squad
    model_dim: int = 64  # Hidden dimension used by experts
    num_experts: int = 4  # Number of experts in this squad
    seq_len: int = 64  # Sequence length for time-series inputs
    features: List[FeatureConfig] = field(default_factory=list)  # Feature configurations
    children: Optional[Dict[str, 'SquadConfig']] = field(default_factory=dict)  # Nested squads
    dropout: float = 0.2  # Dropout rate for regularization

    routing_mode: str = "neural"  # Routing type: neural or deterministic
    dilation_rate: int = 8  # Dilation rate for dilated routing (if applicable)
    routing_feature: str = ""  # Feature key used for routing decisions
    routing_map: dict = field(default_factory=dict)  # Mapping from feature values to experts
    fallback_squad: str = None # fallback squad


@dataclass
class TrainingConfig:
    """Training hyperparameters for an expert model.

    Includes settings for loss computation, optimization, and decision
    thresholds.

    Args:
        pos_weight (float, optional): Weight for positive class in loss.
        alpha (float, optional): Alpha parameter for focal loss.
        gamma (float, optional): Gamma parameter for focal loss.
        decision_threshold (float, optional): Threshold for binary decisions.
        lr (float, optional): Learning rate.
        weight_decay (float, optional): Weight decay for regularization.
        aux_loss_coef (float, optional): Weight for auxiliary losses.
    """
    pos_weight: float = 1.0  # Class imbalance weight for positive samples
    alpha: float = 0.5  # Focal loss alpha parameter
    gamma: float = 2.0  # Focal loss gamma parameter
    decision_threshold: float = 0.5  # Threshold for converting probabilities to binary outputs
    lr: float = 1e-4  # Learning rate for optimizer
    weight_decay: float = 0.5  # Strong L2 regularization to reduce overfitting
    aux_loss_coef: float = 0.05  # Contribution of auxiliary losses to total loss


@dataclass
class ExpertConfig:
    """Top-level configuration for an expert model.

    Combines metadata, squad configurations, and training parameters
    into a single structure for initialization and management.

    Args:
        expert_id (str): Unique identifier for the expert.
        model_dim (int): Internal representation dimension.
        squads (Dict[str, SquadConfig]): Mapping of squad names to configs.
        training (TrainingConfig): Training-related configuration.
    """
    expert_id: str  # Unique identifier for this expert instance
    model_dim: int  # Internal feature dimension
    squads: Dict[str, SquadConfig]  # Squad configurations grouped by name
    training: TrainingConfig  # Training hyperparameters
