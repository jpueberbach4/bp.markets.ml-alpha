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
