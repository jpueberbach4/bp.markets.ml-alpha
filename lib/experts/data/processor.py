import torch
import numpy as np
import logging
from typing import Dict, List, Any
import torch.nn.functional as F

logger = logging.getLogger("DataProcessor")

class MarketDataProcessor:
    """
    Handles the transformation, cleaning, and windowing of raw financial market data.

    This utility class converts raw columnar data from REST APIs into dense, 
    normalized, and temporally aligned PyTorch tensors suitable for sequence models 
    like LSTMs, Transformers, or Hierarchical Mixture of Experts (HMoE).
    """

    @staticmethod
    def columnar_to_tensor(raw_data: Dict[str, List[Any]], target_fields: List[str] = None) -> tuple[torch.Tensor, dict]:
        """
        Converts a dictionary of lists (columnar data) into a unified 2D PyTorch tensor, 
        handling missing values via forward-fill imputation.

        Args:
            raw_data (Dict[str, List[Any]]): A dictionary where keys are feature names and values are chronological data lists.
            target_fields (List[str], optional): A specific list of feature keys to extract. If None, all keys are used. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: A 2D dense tensor of shape [Time, Features]. Returns an empty tensor if parsing fails.
                - dict: A statistics dictionary containing diagnostic info, such as the number of NaNs repaired per column.
        """
        stats = {}
        try:
            # Filter down to the requested feature subset to avoid processing unnecessary data
            if target_fields:
                candidates = [f for f in target_fields if f in raw_data]
            else:
                candidates = sorted(list(raw_data.keys()))

            if not candidates:
                return torch.empty(0), stats

            # Establish a universal temporal boundary to prevent out-of-bounds indexing
            lengths = [len(raw_data[k]) for k in candidates]
            min_len = min(lengths)

            cleaned_columns = []
            final_keys = []

            for k in candidates:
                try:
                    col_data = raw_data[k][:min_len]
                    col = np.array(col_data).astype(np.float32)
                    mask = np.isnan(col)
                    
                    nan_count = np.sum(mask)
                    stats[k] = {'nans_repaired': int(nan_count)}

                    if nan_count > 0:
                        # Implement a vectorized forward-fill (carry-forward) to replace NaNs
                        # This prevents look-ahead bias that would occur with interpolation
                        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
                        np.maximum.accumulate(idx, out=idx)
                        col = col[idx]
                        # Replace any remaining leading NaNs (which have no prior value to carry forward) with 0.0
                        col[np.isnan(col)] = 0.0

                    cleaned_columns.append(col)
                    final_keys.append(k)

                except (ValueError, TypeError):
                    # Skip columns that contain non-numeric string data or corrupted types
                    continue

            if not cleaned_columns:
                return torch.empty(0), stats

            # Stack the isolated 1D arrays into a cohesive 2D matrix
            matrix = np.column_stack(cleaned_columns)
            tensor = torch.from_numpy(matrix).float()
            return tensor, stats

        except Exception as e:
            logger.error(f"Error in dynamic pivoting: {e}")
            return torch.empty(0), stats

    @staticmethod
    def rolling_zscore_1d(col_tensor: torch.Tensor, window_size: int = 180, eps: float = 1e-8) -> torch.Tensor:
        """
        Applies a rolling Z-score normalization to a 1D tensor using 1D average pooling.

        This ensures that financial indicators are normalized relative to their recent 
        historical context, preventing macro-regime shifts from skewing the data distribution.

        Args:
            col_tensor (torch.Tensor): A 1D tensor representing a single time-series feature.
            window_size (int, optional): The lookback period for calculating rolling mean and variance. Defaults to 180.
            eps (float, optional): A small epsilon value to prevent division by zero. Defaults to 1e-8.

        Returns:
            torch.Tensor: The normalized 1D tensor.
        """
        if col_tensor.nelement() == 0 or col_tensor.shape[0] < 2:
            return col_tensor
            
        # Cast to float64 to mitigate catastrophic cancellation errors during variance calculation
        x = col_tensor.unsqueeze(0).unsqueeze(0).to(torch.float64)  
        
        # Pad the left side with replicated initial values so the output maintains identical length
        x_padded = F.pad(x, (window_size - 1, 0), mode='replicate')
        mean = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
        
        x2_padded = F.pad(x**2, (window_size - 1, 0), mode='replicate')
        mean_x2 = F.avg_pool1d(x2_padded, kernel_size=window_size, stride=1)
        
        # Calculate variance using the identity: Var(X) = E[X^2] - (E[X])^2
        var = torch.clamp(mean_x2 - mean**2, min=0.0)
        std = torch.sqrt(var)
        normed = (x - mean) / (std + eps)
        
        return normed.squeeze(0).squeeze(0).to(torch.float32)

    @staticmethod
    def create_windowed_payload(
        features_dict: Dict[str, List[Any]],
        labels_dict: Dict[str, List[Any]],
        feature_fields: List[str],
        label_field: str,
        seq_len: int,
        pre_pivot_fill: int = 0, 
        post_pivot_fill: int = 0,
        features_to_normalize: List[str] = None  
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Orchestrates the complete data transformation pipeline: converting to tensors, normalizing,
        expanding targets, and slicing into sequential batches via sliding windows.

        Args:
            features_dict (Dict[str, List[Any]]): The raw feature dictionary.
            labels_dict (Dict[str, List[Any]]): The raw target label dictionary.
            feature_fields (List[str]): Explicit list of feature names to map.
            label_field (str): The explicit target label name.
            seq_len (int): The sequence length (lookback window) for the sequence models.
            pre_pivot_fill (int, optional): Number of preceding ticks to flag alongside a trigger. Defaults to 0.
            post_pivot_fill (int, optional): Number of trailing ticks to flag alongside a trigger. Defaults to 0.
            features_to_normalize (List[str], optional): List of specific feature names requiring Z-score normalization. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: A 3D feature tensor of shape [Batch, Features, SeqLen].
                - torch.Tensor: A 2D target label tensor of shape [Batch, 1].
        """
        feat_matrix, feat_stats = MarketDataProcessor.columnar_to_tensor(features_dict, target_fields=feature_fields)
        label_matrix, _ = MarketDataProcessor.columnar_to_tensor(labels_dict, target_fields=[label_field])

        if feat_matrix.nelement() > 0:
            logger.info("--- Feature Diagnostics (Pre-Normalization) ---")
            for i, field_name in enumerate(feature_fields):
                col_data = feat_matrix[:, i]
                c_min = col_data.min().item()
                c_max = col_data.max().item()
                c_mean = col_data.mean().item()
                c_std = col_data.std().item()
                nans = feat_stats.get(field_name, {}).get('nans_repaired', 0)
                will_norm = "Yes" if (features_to_normalize and field_name in features_to_normalize) else "No"
                
                logger.info(
                    f"[{field_name}] Nans Fixed: {nans} | "
                    f"Min: {c_min:.4f}, Max: {c_max:.4f} | "
                    f"Mean: {c_mean:.4f}, Std: {c_std:.4f} | "
                    f"Norm: {will_norm}"
                )
            logger.info("-----------------------------------------------")

        # Selectively apply the rolling Z-score algorithm strictly to the explicitly flagged indicators
        if features_to_normalize and feat_matrix.nelement() > 0:
            for i, field_name in enumerate(feature_fields):
                if field_name in features_to_normalize:
                    feat_matrix[:, i] = MarketDataProcessor.rolling_zscore_1d(
                        feat_matrix[:, i], window_size=180
                    )
                    
            logger.info("--- Normalization Diagnostics (Post-Norm) ---")
            for i, field_name in enumerate(feature_fields):
                if field_name in features_to_normalize:
                    col_data = feat_matrix[:, i]
                    logger.info(
                        f"[{field_name}] Norm Min: {col_data.min().item():.4f} | "
                        f"Norm Max: {col_data.max().item():.4f} | "
                        f"Norm Mean: {col_data.mean().item():.4f}"
                    )
            logger.info("---------------------------------------------")

        if feat_matrix.nelement() == 0 or label_matrix.nelement() == 0:
            return torch.empty(0), torch.empty(0)

        # Ensure absolute temporal alignment by truncating to the shortest available array
        min_len = min(len(feat_matrix), len(label_matrix))
        feat_matrix = feat_matrix[:min_len]
        label_matrix = label_matrix[:min_len].view(-1)

        fat_labels = label_matrix.clone()
        pivot_idx = torch.where(label_matrix != 0)[0]
        
        # Expand discrete event triggers into tolerance zones to provide gradient density to the neural network
        for idx in pivot_idx:
            val = label_matrix[idx].item()
            start_idx = max(0, idx - pre_pivot_fill)
            end_idx = min(len(fat_labels) - 1, idx + post_pivot_fill)
            fat_labels[start_idx:end_idx + 1] = val

        fat_labels = fat_labels.view(-1, 1)

        num_samples = min_len - seq_len + 1
        if num_samples <= 0:
            return torch.empty(0), torch.empty(0)

        # Transform the continuous 2D matrix into a 3D batch of sliding sequence windows
        # The unfold operation creates overlapping segments without duplicating physical memory
        feature_windows = feat_matrix.unfold(0, seq_len, 1).transpose(1, 2)
        
        # Shift the target labels forward so the model predicts the label associated with the *end* of the sequence window
        target_labels = fat_labels[seq_len - 1:]

        positive_items = torch.count_nonzero(target_labels).item()
        total_items = target_labels.numel()
        density = (positive_items / total_items) * 100 if total_items > 0 else 0
        
        logger.info("--- Payload Distribution (Strict Multi-Class Zone) ---")
        logger.info(f"Target Label: {label_field}")
        logger.info(f"Positive Windows (!= 0): {int(positive_items)}")
        logger.info(f"Total Windows: {total_items}")
        logger.info(f"Class Density: {density:.4f}%")
        logger.info("------------------------------------------------------")

        return feature_windows, target_labels

    @staticmethod
    def zscore_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Applies a global Z-score normalization across an entire multi-dimensional tensor.

        Warning: Global Z-scoring can introduce look-ahead bias if applied to chronological 
        time-series validation sets. Use rolling Z-scores for strict causal evaluation.

        Args:
            features (torch.Tensor): The raw feature tensor to normalize.
            eps (float, optional): A small epsilon to prevent division by zero. Defaults to 1e-8.

        Returns:
            torch.Tensor: The normalized feature tensor.
        """
        if features.nelement() == 0:
            return features
        mean = features.mean(dim=(0, 1), keepdim=True)
        std = features.std(dim=(0, 1), keepdim=True)
        return (features - mean) / (std + eps)

    @staticmethod
    def rolling_zscore_features(features: torch.Tensor, window_size: int = 180, eps: float = 1e-8) -> torch.Tensor:
        """
        Applies a multi-dimensional rolling Z-score across a batched sequence tensor.

        Args:
            features (torch.Tensor): A multi-dimensional feature tensor. Expected format heavily depends on internal transposition state.
            window_size (int, optional): The historical observation window. Defaults to 180.
            eps (float, optional): A small epsilon to prevent division by zero. Defaults to 1e-8.

        Returns:
            torch.Tensor: The dynamically normalized feature tensor.
        """
        if features.nelement() == 0 or features.shape[0] < 2:
            return features
            
        x = features.T.unsqueeze(0).to(torch.float64)
        x_padded = F.pad(x, (window_size - 1, 0), mode='replicate')
        mean = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
        x2_padded = F.pad(x**2, (window_size - 1, 0), mode='replicate')
        mean_x2 = F.avg_pool1d(x2_padded, kernel_size=window_size, stride=1)
        var = torch.clamp(mean_x2 - mean**2, min=0.0)
        std = torch.sqrt(var)
        normed = (x - mean) / (std + eps)
        
        return normed.squeeze(0).T.to(torch.float32)