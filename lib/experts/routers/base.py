import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseRouter(nn.Module):
    """
    Defines the strict architectural protocol for all routing mechanisms within the Hierarchical Mixture of Experts (HMoE).

    This base class ensures that any implemented router (e.g., Dense, Dilated) strictly adheres to a standard 
    tensor input/output contract. It physically separates the 'macro-routing' decision logic from the 
    'micro-expert' classification logic, allowing for modular hot-swapping of routing strategies without 
    breaking the broader tree topology.
    """

    def __init__(self, input_dim: int, seq_len: int, num_children: int):
        """
        Initializes the routing protocol interface.

        Args:
            input_dim (int): The number of distinct feature channels in the input sequence. 
            seq_len (int): The length of the lookback window (time dimension) for the input sequence.
            num_children (int): The number of downstream nodes (experts or sub-routers) this router must distribute traffic across.
        """
        super().__init__()
        
        # Enforce a minimum dimension of 1 to ensure downstream linear layers do not crash on empty feature sets
        self.input_dim = max(1, input_dim)
        self.seq_len = seq_len
        self.num_children = num_children

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass protocol that every concrete router subclass must implement.

        Args:
            x (torch.Tensor): A 3D sequence tensor strictly formatted as [Batch, Seq_Len, Features].

        Returns:
            torch.Tensor: A 2D tensor of shape [Batch, num_children] containing the un-softmaxed logits, 
                          representing the network's raw routing preferences.

        Raises:
            NotImplementedError: If the inherited class fails to define its specific mathematical routing logic.
        """
        raise NotImplementedError("Every specific router subclass must implement its own forward pass adhering to this protocol.")