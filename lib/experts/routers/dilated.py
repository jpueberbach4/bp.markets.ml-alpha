import torch
import torch.nn as nn
from lib.experts.routers.base import BaseRouter

class DilatedMacroRouter(BaseRouter):
    """
    A low-resolution, sequence-skipping router designed to evaluate the overarching market regime.

    By artificially dilating the input sequence (skipping time steps), this router becomes 
    mathematically blind to high-frequency market noise (e.g., individual candlestick wicks or 
    micro-pullbacks). This forces the network to route traffic based purely on the macro trend, 
    making it the ideal architecture for the root node of a Hierarchical Mixture of Experts.
    """
    
    def __init__(self, input_dim: int, seq_len: int, num_children: int, dilation_rate: int = 4):
        """
        Initializes the DilatedMacroRouter network architecture.

        Args:
            input_dim (int): The number of distinct feature channels per time step.
            seq_len (int): The total continuous length of the raw input window.
            num_children (int): The number of downstream nodes to route traffic to.
            dilation_rate (int, optional): The step size for sampling the sequence. A rate of 4 
                                           means the router only looks at every 4th candle. Defaults to 4.
        """
        super().__init__(input_dim, seq_len, num_children)
        self.dilation_rate = dilation_rate
        
        # Calculate the exact footprint of the reduced sequence to size the input linear layer correctly
        self.dilated_seq_len = len(range(0, seq_len, dilation_rate))
        
        # Because the sequence length is artificially reduced, this router requires significantly 
        # fewer parameters than a DenseRouter, making it highly computationally efficient.
        self.net = nn.Sequential(
            nn.Linear(self.input_dim * self.dilated_seq_len, 128),
            nn.GELU(),
            nn.Linear(128, num_children)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the dilated forward routing pass.

        Args:
            x (torch.Tensor): A 3D continuous sequence tensor of shape [Batch, Seq_Len, Features].

        Returns:
            torch.Tensor: A 2D tensor of shape [Batch, num_children] containing the raw routing 
                          logits for each downstream macro-squad.
        """
        # Slice the temporal dimension using the dilation rate to skip intermediate ticks.
        # Shape transition: [Batch, Seq_Len, Features] -> [Batch, Dilated_Seq_Len, Features]
        x_dilated = x[:, ::self.dilation_rate, :] 
        
        # Flatten the compressed macro-sequence into a 1D vector for the MLP.
        x_flat = x_dilated.reshape(x_dilated.size(0), -1)
        
        return self.net(x_flat)