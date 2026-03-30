import torch
import torch.nn as nn

from lib.experts.routers.base import BaseRouter

class DenseRouter(BaseRouter):
    """
    A high-resolution, sequence-flattening router designed to analyze every individual time step.

    Unlike dilated routers that skip temporal steps to understand macro-trends, the DenseRouter 
    flattens the entire sequence window (Time x Features) into a single 1D vector. This allows 
    the Multi-Layer Perceptron (MLP) to detect highly specific, localized micro-patterns 
    (e.g., exact candlestick formations) at the cost of a higher parameter count.
    """
    
    def __init__(self, input_dim: int, seq_len: int, num_children: int):
        """
        Initializes the DenseRouter network architecture.

        Args:
            input_dim (int): The number of distinct feature channels per time step.
            seq_len (int): The total number of consecutive time steps in the input window.
            num_children (int): The number of downstream experts or nodes to route traffic to.
        """
        super().__init__(input_dim, seq_len, num_children)

        # The input layer must accommodate every feature across every time step simultaneously.
        # This brute-force flattening guarantees no data point is skipped, making it ideal 
        # for 'Leaf' routers that need to pull the trigger on precise, immediate setups.
        self.net = nn.Sequential(
            nn.Linear(self.input_dim * seq_len, 128),
            nn.GELU(),
            nn.Linear(128, num_children)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward routing pass.

        Args:
            x (torch.Tensor): A 3D sequence tensor of shape [Batch, Seq_Len, Features].

        Returns:
            torch.Tensor: A 2D tensor of shape [Batch, num_children] containing the raw routing 
                          logits (un-normalized probabilities) for each downstream node.
        """
        # Flatten the sequence and feature dimensions into a single contiguous vector per batch sample.
        # Shape transition: [Batch, Seq_Len, Features] -> [Batch, Seq_Len * Features]
        x_flat = x.reshape(x.size(0), -1)
        
        return self.net(x_flat)