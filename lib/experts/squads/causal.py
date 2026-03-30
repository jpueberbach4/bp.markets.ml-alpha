import torch
import torch.nn as nn
import torch.nn.functional as F
from tutel import moe as tutel_moe
from lib.experts.squads.registry import SquadRegistry
from lib.experts.config import SquadConfig


class CausalConv1d(nn.Module):
    """
    A depthwise separable 1D convolution layer mathematically constrained to be strictly causal.

    In financial time-series forecasting, standard convolutions suffer from 'look-ahead bias' 
    because their kernels aggregate data from both the past and the future relative to the center 
    index. This layer solves that by asymmetrically padding the input sequence on the left side, 
    ensuring that the computation at time step 't' can only access data from time '<= t'.

    Furthermore, it utilizes a depthwise separable architecture (splitting the spatial and 
    cross-channel computations) to drastically reduce parameter count and prevent overfitting 
    on noisy financial features.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        """
        Initializes the CausalConv1d layer.

        Args:
            in_channels (int): Number of distinct feature channels in the input sequence.
            out_channels (int): Number of feature channels produced by the pointwise mixing.
            kernel_size (int): The spatial reach of the temporal filter.
            dilation (int, optional): The spacing between kernel elements, used to expand the 
                                      receptive field without increasing parameter count. Defaults to 1.
        """
        super().__init__()

        # Calculate the exact amount of asymmetric left-padding required to shift the entire
        # convolutional output forward in time, physically preventing access to future indices.
        self.pad_size = (kernel_size - 1) * dilation

        # Step 1: Spatial Filtering (Depthwise)
        # Groups=in_channels forces the network to apply a separate 1D filter to each feature 
        # channel independently, learning temporal patterns without mixing the features yet.
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=self.pad_size,
            dilation=dilation,
            groups=in_channels
        )

        # Step 2: Feature Mixing (Pointwise)
        # A 1x1 convolution acts as a dense linear layer across the channel dimension, 
        # combining the independently filtered temporal patterns into the final output space.
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the causal convolution operation.

        Args:
            x (torch.Tensor): A 3D sequence tensor of shape [Batch, Channels, Seq_Len].

        Returns:
            torch.Tensor: The causally filtered output sequence of shape [Batch, Out_Channels, Seq_Len].
        """
        x = self.depthwise(x)
        
        # Enforce strict causality by surgically slicing off the right-most temporal steps.
        # Because we artificially padded the left side by 'self.pad_size', the convolution 
        # inherently calculated 'self.pad_size' extra steps into the future. Removing them 
        # realigns the sequence to match the original input length while preserving the causal shift.
        x = x[:, :, :-self.pad_size]

        return self.pointwise(x)


class ResidualCausalBlock(nn.Module):
    """
    A foundational building block for temporal architectures, combining a causal convolution 
    with a residual skip connection.

    By wrapping the CausalConv1d layer in a residual structure, this block allows gradients 
    to flow unhindered during backpropagation. This prevents the vanishing gradient problem 
    in deep temporal networks, allowing the model to safely bypass the convolution if the 
    transformation is deemed unnecessary for the current routing decision.
    """

    def __init__(self, dim: int, kernel_size: int, dilation: int):
        """
        Initializes the ResidualCausalBlock.

        Args:
            dim (int): The number of feature channels. This block enforces identical input/output 
                       dimensions to support the residual addition.
            kernel_size (int): The temporal footprint of the internal causal convolution.
            dilation (int): The dilation factor for expanding the receptive field.
        """
        super().__init__()

        self.conv = CausalConv1d(dim, dim, kernel_size, dilation=dilation)
        self.layer_norm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the residual block.

        Args:
            x (torch.Tensor): A 3D sequence tensor of shape [Batch, Channels, Seq_Len].

        Returns:
            torch.Tensor: The activated and normalized sequence tensor, combined with the residual identity.
        """
        residual = x

        x = self.conv(x)

        # PyTorch LayerNorm expects the normalized dimension (Channels) to be the absolute last dimension.
        # We must temporarily transpose from [Batch, Channels, Seq_Len] to [Batch, Seq_Len, Channels],
        # apply the normalization, and safely transpose back to preserve CNN compatibility.
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        x = self.gelu(x)

        # The residual skip connection creates a direct mathematical highway from input to output.
        return x + residual