import torch
from torch.utils.data import Dataset


class SquadDataset(Dataset):
    """
    A custom PyTorch Dataset designed to serve structurally partitioned feature tensors to a Hierarchical Mixture of Experts (HMoE).

    Unlike standard datasets that yield a single unified feature tensor, this dataset maintains the 
    topological boundaries of the network. It stores and yields data as a dictionary where each key 
    represents a specific neural node (squad/expert), and the value is the exact slice of features 
    that specific node requires. This allows the data loader to natively feed the tree topology 
    without requiring the model to slice tensors internally during the forward pass.
    """

    def __init__(self, squad_tensors: dict, labels: torch.Tensor):
        """
        Initializes the SquadDataset with pre-partitioned memory tensors.

        Args:
            squad_tensors (dict): A dictionary mapping node paths (e.g., 'root.detect_tops.rsi') 
                                  to their respective sequence tensors. Expected tensor shape is 
                                  typically [Samples, Channels, Seq_Len].
            labels (torch.Tensor): A tensor containing the aligned ground-truth target labels, 
                                   typically of shape [Samples, ...].
        """
        self.squad_tensors = squad_tensors
        self.labels = labels
        
        # Assume all partitioned tensors share the same primary temporal dimension
        self.length = labels.shape[0]

    def __len__(self):
        """
        Calculates the total number of aligned sequence windows available in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves a single, temporally aligned sample across all partitioned network nodes.

        By utilizing a dictionary comprehension, this method dynamically slices the requested 
        temporal index from every node's dedicated feature tensor simultaneously, guaranteeing 
        that the entire HMoE receives data from the exact same point in time.

        Args:
            idx (int): The absolute temporal index of the sequence window to retrieve.

        Returns:
            tuple: A tuple containing:
                - dict: The partitioned feature slices mapped to their respective node identifiers.
                - torch.Tensor: The ground-truth label corresponding to this specific sequence window.
        """
        features = {
            squad_name: tensor[idx]
            for squad_name, tensor in self.squad_tensors.items()
        }
        
        return features, self.labels[idx]