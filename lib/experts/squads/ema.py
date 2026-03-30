import torch
import torch.nn as nn
import torch.nn.functional as F
from tutel import moe as tutel_moe
from lib.experts.squads.registry import SquadRegistry
from lib.experts.config import SquadConfig
from lib.experts.squads.causal import CausalConv1d, ResidualCausalBlock


@SquadRegistry.register('ema')
class EmaSquad(nn.Module):
    """
    A specialized neural squad designed to evaluate market momentum and structural trends 
    using Exponential Moving Average (EMA) ribbon features.

    Unlike raw price action, EMA ribbons represent smoothed, lagging momentum data. To capture 
    the compression, expansion, and twisting of these ribbons, this module utilizes a wider 
    convolutional kernel (size 5) and a progressively dilated residual stack to establish a 
    much deeper receptive field than the micro-pattern squads. The resulting representation 
    is routed through a localized Mixture-of-Experts (MoE) block.
    """

    def __init__(self, config: SquadConfig):
        """
        Initializes the EmaSquad architecture.

        Args:
            config (SquadConfig): The configuration object defining input dimensions, 
                                  hidden model dimensions, and MoE expert capacities.
        """
        super().__init__()

        self.F = config.input_dim
        self.M = config.model_dim

        # The initial projection uses a wider kernel_size=5 to immediately capture a larger 
        # chunk of the slowly moving EMA curves, projecting them into the hidden model space.
        self.input_projection = CausalConv1d(
            in_channels=self.F,
            out_channels=self.M,
            kernel_size=5,
            dilation=1
        )

        # A deeper stack of dilated residual blocks exponentially increases the temporal receptive field.
        # This allows the network to "look back" far enough to understand if the EMA ribbon has been 
        # compressing over the medium-term or if it is currently over-extended.
        self.dilated_stack = nn.ModuleList([
            ResidualCausalBlock(self.M, kernel_size=5, dilation=1),
            ResidualCausalBlock(self.M, kernel_size=5, dilation=2),
            ResidualCausalBlock(self.M, kernel_size=5, dilation=4),
            # Deeper historical contexts (dilation=8, 16) are commented out but available if the 
            # network requires a broader understanding of deep structural memory.
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=8),
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=16),
        ])

        self.norm = nn.LayerNorm(self.M)

        # The specialized MoE layer routes the momentum state to the appropriate expert.
        # Top-2 routing ensures that the network blends the opinions of multiple experts 
        # when evaluating ambiguous momentum states (e.g., during a ribbon twist).
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
        Processes a batched sequence of EMA features through the causal stack and MoE layer.

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
        # Pass the input through the causal projection and dilated residual stack to extract 
        # the momentum structure without leaking future data points.
        x = self.input_projection(x)

        for block in self.dilated_stack:
            x = block(x)

        # Terminal State Isolation
        # Following the convolution stack, the tensor maintains its [Batch, Channels, Seq_Len] shape.
        # We slice out only the absolute final time step to capture the cumulative historical context 
        # representing the "present" market structure.
        terminal_state = x[:, :, -1]

        # Expert Routing and Computation
        # Normalize the terminal state vector before distributing it to the specialized FFN experts.
        tokens = self.norm(terminal_state)
        moe_out = self.moe_layer(tokens)
        
        return moe_out