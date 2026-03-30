import torch
import torch.nn as nn
from tutel import moe as tutel_moe
from lib.experts.squads.registry import SquadRegistry
from lib.experts.config import SquadConfig
from lib.experts.squads.causal import CausalConv1d, ResidualCausalBlock


@SquadRegistry.register('rsi')
class RSISquad(nn.Module):
    """
    A specialized neural squad engineered to extract mean-reversion and momentum divergence 
    signals from Relative Strength Index (RSI) features.

    Because the RSI features fed into this squad are heavily pre-smoothed during data ingestion, 
    the raw inputs already contain substantial macro-momentum context. Consequently, this module 
    employs a deliberately shallow causal residual stack. It relies on the pre-smoothing for 
    macro-awareness while dedicating its internal parameter capacity to identifying immediate, 
    high-conviction micro-divergences before routing to the Mixture-of-Experts layer.
    """

    def __init__(self, config: SquadConfig):
        """
        Initializes the RSISquad architecture.

        Args:
            config (SquadConfig): The configuration object defining input dimensions, 
                                  hidden model dimensions, and MoE expert capacities.
        """
        super().__init__()

        self.F = config.input_dim
        self.M = config.model_dim

        # The initial causal projection maps the raw, smoothed RSI oscillator data 
        # into the latent model dimension. A kernel size of 5 gives the network an 
        # immediate context window of the last 5 time steps.
        self.input_projection = CausalConv1d(
            in_channels=self.F,
            out_channels=self.M,
            kernel_size=5,
            dilation=1
        )

        # The residual stack defines the temporal receptive field.
        # Because the input RSI streams are already smoothed (effectively encoding long-term 
        # moving averages), large dilations (4, 8, 16) are mathematically redundant and risk 
        # over-smoothing the signal. The stack is strictly limited to short and medium-short 
        # context (dilation 1 and 2) to capture immediate momentum fractures.
        self.dilated_stack = nn.ModuleList([
            ResidualCausalBlock(self.M, kernel_size=5, dilation=1),
            ResidualCausalBlock(self.M, kernel_size=5, dilation=2),
            # Deeper blocks are explicitly disabled to prevent signal degradation on pre-smoothed data.
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=4),
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=8),
            #ResidualCausalBlock(self.M, kernel_size=5, dilation=16),
        ])

        self.norm = nn.LayerNorm(self.M)

        # The terminal representation is routed dynamically through a Top-2 Mixture-of-Experts layer.
        # Note the explicitly higher gate_noise (0.2). This forces the router to aggressively 
        # explore different experts during training, preventing the highly normalized RSI features 
        # from collapsing into a single, lazy "echo chamber" expert.
        self.moe_layer = tutel_moe.moe_layer(
            gate_type={
                'type': 'top',
                'k': 2,
                'capacity_factor': 1.5,
                'gate_noise': 0.2
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
        Processes a batched sequence of RSI features through the causal stack and MoE layer.

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
        # Safely convolute the RSI data across the temporal axis without leaking future steps.
        x = self.input_projection(x)

        for block in self.dilated_stack:
            x = block(x)

        # Terminal State Isolation
        # Following the convolutions, the tensor shape is [Batch, Channels, Seq_Len].
        # Because we are evaluating the current market momentum, we aggressively slice out 
        # the final index (t=0 relative to the present), discarding the historical step representations.
        terminal_state = x[:, :, -1]

        # Expert Routing and Computation
        # Normalize the isolated terminal vector and dispatch it to the MoE block for final classification.
        tokens = self.norm(terminal_state)
        moe_out = self.moe_layer(tokens)
        
        return moe_out