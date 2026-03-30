import torch
import torch.nn as nn
import torch.nn.functional as F
from tutel import moe as tutel_moe
from lib.experts.squads.registry import SquadRegistry
from lib.experts.config import SquadConfig
from lib.experts.squads.causal import CausalConv1d, ResidualCausalBlock


@SquadRegistry.register('indicator')
class IndicatorSquad(nn.Module):
    """
    A versatile neural squad engineered to extract temporal features from generic market indicators 
    (e.g., ADX, ATR, MACD).

    This module utilizes causal convolutions to safely evaluate sequences of indicator data without 
    incurring look-ahead bias. The current architectural configuration employs a relatively shallow 
    residual stack, intentionally forcing the network to specialize in short-term indicator patterns 
    rather than deep historical contexts, before routing the representation to a Mixture-of-Experts layer.
    """

    def __init__(self, config: SquadConfig):
        """
        Initializes the IndicatorSquad architecture.

        Args:
            config (SquadConfig): The configuration object defining input dimensions, 
                                  hidden model dimensions, and MoE expert capacities.
        """
        super().__init__() 

        self.F = config.input_dim
        self.M = config.model_dim

        # The initial causal projection maps the raw indicator signals into the latent model dimension.
        # A kernel size of 5 gives the network an immediate window into the last 5 time steps of the indicator.
        self.input_projection = CausalConv1d(
            in_channels=self.F,
            out_channels=self.M,
            kernel_size=5,
            dilation=1
        )

        # The residual stack defines the temporal receptive field.
        # Currently, only the baseline block (dilation=1) is active. This configuration signals that 
        # the network should prioritize immediate, short-term structural shifts in the indicators.
        # Deeper temporal memory blocks are retained in the architecture but commented out, allowing 
        # for rapid activation if the model requires deeper historical context during future tuning.
        self.dilated_stack = nn.ModuleList([
            ResidualCausalBlock(self.M, kernel_size=5, dilation=1),
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=2),
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=4),
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=8),
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=16),
        ])

        self.norm = nn.LayerNorm(self.M)

        # The terminal representation is routed dynamically through a Top-2 Mixture-of-Experts layer.
        # This allows the network to spin up specific experts for distinct indicator regimes 
        # (e.g., one expert for high-volatility ADX environments, another for low-volatility chop).
        self.moe_layer = tutel_moe.moe_layer(
            gate_type={
                'type': 'top',
                'k': 2,
                'capacity_factor': 1.5
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
        Processes a batched sequence of indicator features through the causal stack and MoE layer.

        Args:
            x (torch.Tensor): The input sequence tensor. Expected shape is either 
                              [Batch, Seq_Len, Features] or [Batch, Features, Seq_Len].

        Returns:
            torch.Tensor: The final feature representation of the terminal state, 
                          post-MoE routing, with shape [Batch, Model_Dim].
        """
        # Dimensionality Alignment for 1D Convolutions
        # PyTorch Conv1d strictly requires the channel (feature) dimension to precede the sequence dimension.
        # If the incoming tensor is structured as [Batch, Seq_Len, Features], safely transpose it.
        if x.dim() == 3 and x.size(2) == self.F:
            x = x.permute(0, 2, 1).contiguous()

        # Causal Feature Extraction
        # Safely convolute the indicator data across the temporal axis without leaking future steps.
        x = self.input_projection(x)

        for block in self.dilated_stack:
            x = block(x)

        # Terminal State Isolation
        # Following the convolutions, the tensor shape is [Batch, Channels, Seq_Len].
        # Because we are evaluating the current market trigger point, we aggressively slice out 
        # the final index (t=0 relative to the present), discarding the historical step representations.
        terminal_state = x[:, :, -1]

        # Expert Routing and Computation
        # Normalize the isolated terminal vector and dispatch it to the MoE block for final classification.
        tokens = self.norm(terminal_state)
        moe_out = self.moe_layer(tokens)
        
        return moe_out