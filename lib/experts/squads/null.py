import torch
import torch.nn as nn
from tutel import moe as tutel_moe
from lib.experts.squads.registry import SquadRegistry
from lib.experts.config import SquadConfig
from lib.experts.squads.causal import CausalConv1d, ResidualCausalBlock


@SquadRegistry.register('null')
class NullSquad(nn.Module):
    """
    An architectural safety valve and computational sinkhole for the MoE routing tree.

    In a sparse Mixture-of-Experts architecture, routers occasionally need to discard traffic 
    when the market state is completely ambiguous (e.g., severe sideways chop). The NullSquad 
    acts as a stable structural node to absorb this traffic. If assigned zero local features, 
    it provides a mathematically valid pass-through to satisfy the network's gradient requirements 
    without crashing. If features are provided, it functions as a lightweight linear expert.
    """
    
    def __init__(self, config: SquadConfig):
        """
        Initializes the NullSquad architecture.

        Args:
            config (SquadConfig): The configuration object defining input dimensions and 
                                  hidden model dimensions.
        """
        super().__init__()
        self.config = config

        # Route 1: The Computational Sinkhole (0 Features)
        # If the config explicitly passes no features, we must still satisfy PyTorch's 
        # dimensional requirements for backpropagation. We construct a minimal sequential 
        # network that expects a dummy tensor of shape [Batch, 1], projecting it safely 
        # into the required model dimension.
        if config.input_dim == 0:
            self.net = nn.Sequential(
                nn.Linear(1, config.model_dim),
                nn.GELU(),
                nn.Linear(config.model_dim, config.model_dim)
            )
            
        # Route 2: The Baseline Linear Expert (>0 Features)
        # If features are assigned to this node, it abandons the sinkhole behavior and 
        # acts as a standard, non-temporal linear expert. It uses a basic MLP without 
        # causal convolutions, prioritizing extreme computational speed over pattern depth.
        else:
            self.net = nn.Sequential(
                nn.Linear(config.input_dim, config.model_dim),
                nn.GELU(),
                nn.LayerNorm(config.model_dim),
                nn.Linear(config.model_dim, config.model_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the incoming tensor through the fallback network.

        Args:
            x (torch.Tensor): The input data. If input_dim == 0, this will be a dummy tensor 
                              of shape [Batch, 1]. If >0, it expects [Batch, Features]. 
                              If a 3D sequence tensor slips through, it dynamically compresses it.

        Returns:
            torch.Tensor: The projected output tensor of shape [Batch, Model_Dim].
        """
        # Architectural Failsafe for Sequence Tensors
        # If upstream nodes bypass the expected 2D formatting and pass a full 3D sequence 
        # [Batch, Seq_Len, Features], a standard Linear layer will crash. This block catches 
        # that dimension mismatch and smoothly compresses the temporal axis by averaging the 
        # last two time steps, converting it safely back to [Batch, Features].
        if x.dim() == 3:
             smooth_window = min(2, x.size(1))
             x = x[:, -smooth_window:, :].mean(dim=1)
             
        return self.net(x)