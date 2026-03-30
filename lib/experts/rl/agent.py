import os
import torch
import logging
import uuid

# Logger dedicated to RL agent operations
logger = logging.getLogger("RLAgent")


class RLAgent:
    """Baseline reinforcement learning agent wrapper for MoE inference.

    Despite its name, this agent does not currently perform any learning.
    It acts as a deterministic inference layer over a trained Mixture-of-Experts model.

    Responsibilities:
    - Run forward inference using the underlying router network
    - Apply a threshold to convert probabilities into actions
    - Track decisions via unique cohort IDs for downstream trade management
    - Load model checkpoints and associated thresholds

    This structure allows easy extension into a true RL agent in the future.
    """

    def __init__(self, router_network, device='cpu', manual_threshold=0.27):
        """Initializes the RLAgent.

        Args:
            router_network (torch.nn.Module): Trained MoE model used for inference.
            device (str or torch.device, optional): Execution device. Defaults to 'cpu'.
            manual_threshold (float, optional): Decision threshold for binary action.
                Defaults to 0.27.
        """
        # Store reference to model and move it to the desired device
        self.router = router_network.to(device)

        # Device used for inference
        self.device = device

        # Threshold used to convert probability outputs into binary actions
        self.manual_threshold = manual_threshold

        # Tracks active decisions keyed by cohort ID
        # This enables future RL updates or lifecycle tracking
        self.active_decisions = {}

    def select_expert(self, state_dict):
        """Performs inference and selects an action based on model output.

        The model produces a probability score, which is compared against
        a threshold to determine whether to trigger an action.

        Args:
            state_dict (dict): Input payload containing model features.

        Returns:
            tuple:
                - action_val (int): Binary decision (1 = act, 0 = no action)
                - cohort_id (str): Unique identifier for tracking this decision
        """
        with torch.no_grad():
            # Perform forward pass through the MoE model
            predictions, _ = self.router.forward_and_loss(
                state_dict,
                compute_loss=False
            )

            # Extract scalar probability value
            prob_val = predictions[0, 0].item()

            # Convert probability into binary decision using threshold
            action_val = 1 if prob_val >= self.manual_threshold else 0

        # Generate a unique identifier for this decision
        cohort_id = str(uuid.uuid4())

        # Store decision for potential future updates or tracking
        self.active_decisions[cohort_id] = {
            'action': action_val
        }

        return action_val, cohort_id

    def update_from_pnl(self, cohort_id, realized_pnl):
        """Handles post-trade updates (currently no-op).

        This method is a placeholder for reinforcement learning updates.
        In a full RL implementation, this is where:
        - Rewards would be computed
        - Policy/value updates would occur
        - Credit assignment would be handled

        Currently, it only cleans up tracked decisions.

        Args:
            cohort_id (str): Identifier of the decision being updated.
            realized_pnl (float): Realized profit/loss (unused).

        Returns:
            None
        """
        # Remove decision from active tracking if it exists
        if cohort_id in self.active_decisions:
            self.active_decisions.pop(cohort_id)

        # No learning is performed in this baseline implementation
        pass

    def load_checkpoint(self, checkpoint_path):
        """Loads model weights and retrieves associated decision threshold.

        Args:
            checkpoint_path (str): Path to the saved checkpoint file.

        Returns:
            float: Loaded decision threshold if present, otherwise current threshold.
        """
        # Validate checkpoint path
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return self.manual_threshold

        # Load checkpoint onto correct device
        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Extract model weights (supports both raw state_dict and wrapped checkpoints)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # Load weights into router network
        self.router.load_state_dict(state_dict)

        # Return stored threshold if available, otherwise fallback to current value
        return ckpt.get('best_threshold', self.manual_threshold)