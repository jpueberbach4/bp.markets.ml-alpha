import torch
import torch.nn as nn
from tutel import moe as tutel_moe
from lib.experts.squads.registry import SquadRegistry
from lib.experts.config import SquadConfig
from lib.experts.squads.causal import CausalConv1d, ResidualCausalBlock


@SquadRegistry.register('candlestick')
class CandlestickSquad(nn.Module):
    """
    A specialized neural squad dedicated to extracting micro-patterns from raw candlestick data.

    This module utilizes a stack of strictly causal 1D convolutions to process time-series data. 
    By enforcing causality, the architecture guarantees that the mathematical representation of 
    a given time step is completely blind to future steps, preventing data leakage. The resulting 
    feature representation is then routed through a localized Mixture-of-Experts (MoE) block.
    """

    def __init__(self, config: SquadConfig):
        """
        Initializes the CandlestickSquad architecture.

        Args:
            config (SquadConfig): The configuration object defining input dimensions, 
                                  hidden model dimensions, and MoE expert capacities.
        """
        super().__init__() 

        self.F = config.input_dim
        self.M = config.model_dim

        # The initial projection layer transforms raw input features into the hidden model dimension.
        # A causal convolution with dilation=1 establishes the baseline chronological sliding window.
        self.input_projection = CausalConv1d(
            in_channels=self.F,
            out_channels=self.M,
            kernel_size=3,
            dilation=1
        )

        # A shallow stack of residual causal blocks expands the receptive field slightly.
        # Unlike macro-trend routers, this stack is intentionally kept shallow to force the 
        # linear matrices to specialize in short-term, micro-price action (e.g., 3-5 candle patterns).
        self.dilated_stack = nn.ModuleList([
            ResidualCausalBlock(self.M, kernel_size=3, dilation=1), 
            ResidualCausalBlock(self.M, kernel_size=3, dilation=2), 
        ])

        self.norm = nn.LayerNorm(self.M)

        # The specialized MoE layer handles the final feature extraction and decision routing.
        # It uses a Top-2 gating mechanism with capacity limits to enforce load balancing 
        # across the discrete experts.
        self.moe_layer = tutel_moe.moe_layer(
            gate_type={
                'type': 'top',
                'k': 2,
                'capacity_factor': 1.5,
                'gate_noise': 0.1
            }, 
            model_dim=self.M,
            experts={
                'type': 'ffn',
                'hidden_size_per_expert': self.M * 2,
                'activation_fn': nn.GELU(), 
                'count_per_node': getattr(config, 'num_experts', 4),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a batched sequence of candlestick features through the causal stack and MoE layer.

        Args:
            x (torch.Tensor): The input sequence tensor. Expected shape is either 
                              [Batch, Seq_Len, Features] or [Batch, Features, Seq_Len].

        Returns:
            torch.Tensor: The final feature representation of the terminal state, 
                          post-MoE routing, with shape [Batch, Model_Dim].
        """
        # Dimensionality Alignment for 1D Convolutions
        # PyTorch Conv1d expects the channel (feature) dimension to precede the sequence dimension.
        # If the input arrives as [Batch, Seq_Len, Features], safely transpose it to [Batch, Features, Seq_Len].
        if x.dim() == 3 and x.size(2) == self.F:
            x = x.permute(0, 2, 1).contiguous()

        # Causal Feature Extraction
        # Project the raw inputs into the latent dimensional space while preserving strict temporal ordering.
        x = self.input_projection(x)

        for block in self.dilated_stack:
            x = block(x)

        # Terminal State Isolation
        # Following the convolution stack, the tensor maintains its [Batch, Channels, Seq_Len] shape.
        # Since this network triggers decisions based on current market conditions, we slice out 
        # only the absolute final time step representing the "present" context.
        terminal_state = x[:, :, -1]

        # 4. Expert Routing and Computation
        # Normalize the terminal state vector before distributing it to the specialized FFN experts.
        tokens = self.norm(terminal_state)
        moe_out = self.moe_layer(tokens)
        
        return moe_out