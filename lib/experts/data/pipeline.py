import torch
import numpy as np
from datetime import datetime, timezone
import logging

from lib.experts.data.types.rest import RESTMarketFetcher
from lib.experts.data.processor import MarketDataProcessor
import config

logger = logging.getLogger("DataPipeline")


class FastDictDataLoader:
    """
    A lightweight, high-performance custom data loader optimized for dictionary-based tensor payloads.

    This loader bypasses the overhead of standard PyTorch DataLoaders by keeping all tensors 
    pre-loaded in VRAM (or RAM) and yielding sliced dictionary batches. It also handles 
    on-the-fly data augmentation via dynamic Gaussian noise injection.
    """

    def __init__(
        self,
        squad_tensors: dict[str, torch.Tensor],
        Y: torch.Tensor,
        batch_size: int,
        is_training: bool = False,
        noise_map: dict[str, torch.Tensor] = None
    ):
        """
        Initializes the FastDictDataLoader.

        Args:
            squad_tensors (dict[str, torch.Tensor]): A dictionary mapping node paths to their respective feature tensors.
            Y (torch.Tensor): The target label tensor aligned with the feature tensors.
            batch_size (int): The number of samples to yield per batch.
            is_training (bool, optional): If True, enables batch shuffling and noise injection. Defaults to False.
            noise_map (dict[str, torch.Tensor], optional): A dictionary mapping node paths to a tensor of standard deviations for noise injection. Defaults to None.
        """
        self.squad_tensors = squad_tensors
        self.Y = Y
        self.batch_size = batch_size
        self.is_training = is_training
        
        # Default to an empty dictionary if no noise map is provided to prevent NoneType errors during iteration
        self.noise_map = noise_map or {}

        self.dataset_len = len(Y)
        self.device = Y.device

    def __iter__(self):
        """
        Yields batches of features and labels, applying dynamic perturbations if configured.

        Yields:
            dict: A payload dictionary containing 'features' (dict of batched tensors) and 'labels' (batched target tensor).
        """
        # Generate randomized indices for training to prevent cyclical learning bias, otherwise maintain chronological order
        if self.is_training:
            indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            indices = torch.arange(self.dataset_len, device=self.device)

        for start_idx in range(0, self.dataset_len, self.batch_size):
            batch_idx = indices[start_idx:start_idx + self.batch_size]

            batch_features = {}
            for k, v in self.squad_tensors.items():
                batch_tensor = v[batch_idx]
                
                # Apply data augmentation only during active training passes for nodes with defined noise thresholds
                if self.is_training and k in self.noise_map:
                    noise_std = self.noise_map[k]
                    
                    if noise_std is not None:
                         # Generate a fresh noise tensor matching the batch shape. 
                         # The noise_std tensor relies on PyTorch broadcasting semantics to apply specific scalar deviations across the feature dimension.
                         noise = torch.randn_like(batch_tensor) * noise_std
                         batch_tensor = batch_tensor + noise
                         
                batch_features[k] = batch_tensor

            yield {
                'features': batch_features,
                'labels': {'reversal_signal': self.Y[batch_idx]}
            }

    def __len__(self):
        """
        Calculates the total number of batches in the dataset.

        Returns:
            int: The total number of batches, using ceiling division to account for partial final batches.
        """
        # Utilize ceiling division to ensure the final partial batch is accounted for in the length calculation
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


class MoEDataPipeline:
    """
    Orchestrates the ingestion, processing, and formatting of raw market data into structure-aware tensors.

    This pipeline acts as the bridge between raw REST API data and the Hierarchical Mixture of Experts, 
    translating complex nested configurations into aligned, windowed tensor batches.
    """

    def __init__(self, base_url="http://localhost:8000"):
        """
        Initializes the MoEDataPipeline with its respective fetcher and processor utilities.

        Args:
            base_url (str, optional): The base URL for the REST market data API. Defaults to "http://localhost:8000".
        """
        self.fetcher = RESTMarketFetcher(base_url)
        self.processor = MarketDataProcessor()
        self.last_squad_map = {}

    @staticmethod
    def date_str_to_epoch_ms(date_str: str) -> int:
        """
        Converts a standard YYYY-MM-DD date string into UTC epoch milliseconds.

        Args:
            date_str (str): The date string to convert.

        Returns:
            int: The equivalent UTC epoch timestamp in milliseconds.
        """
        # Force the datetime object into UTC to prevent local machine timezone settings from shifting the data window
        return int(
            datetime.strptime(date_str, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp() * 1000
        )

    def _flatten_squads(self, squads_dict, prefix="") -> dict:
        """
        Recursively traverses a hierarchical squad configuration and flattens it into a 1D mapping.

        Args:
            squads_dict (dict): The nested dictionary of SquadConfig objects.
            prefix (str, optional): The accumulated dot-notation path of the current node. Defaults to "".

        Returns:
            dict: A flattened dictionary where keys are node paths (e.g., 'root.detect_tops.rsi') and values are feature lists.
        """
        flat = {}

        for k, v in squads_dict.items():
            # Construct the absolute path identifier for the current node using dot notation
            path = f"{prefix}.{k}" if prefix else k

            if v.features:
                flat[path] = v.features

            # Recursively explore and append children nodes, passing down the current path as the new prefix
            if v.children:
                flat.update(self._flatten_squads(v.children, path))

        return flat

    def fetch_squads_and_window(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        squads_config: dict,
        target_feature: str,
        include_ohlcv: bool = False
    ):
        """
        Fetches raw market data, processes it into a sliding window format, and maps the resulting tensors to specific network nodes.

        Args:
            symbol (str): The market ticker symbol (e.g., 'GBP-USD').
            timeframe (str): The candle timeframe (e.g., '4h').
            start_date (str): The start date boundary (YYYY-MM-DD).
            end_date (str): The end date boundary (YYYY-MM-DD).
            squads_config (dict): The hierarchical dictionary defining the MoE structure and required features.
            target_feature (str): The name of the specific feature column to act as the ground-truth label.
            include_ohlcv (bool, optional): If True, fetches and aligns raw OHLCV data for visualization purposes. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - dict: squad_tensors mapping node paths to sequence tensors.
                - torch.Tensor: Y labels tensor.
                - dict: squad_noise_maps mapping node paths to standard deviation tensors.
                - dict (optional): aligned_ohlcv dictionary, returned only if include_ohlcv is True.
        """
        after_ms = self.date_str_to_epoch_ms(start_date)
        until_ms = self.date_str_to_epoch_ms(end_date)

        merged_features = {}
        flat_squads = self._flatten_squads(squads_config)

        squad_actual_keys = {node_path: [] for node_path in flat_squads.keys()}
        features_to_normalize = []
        feature_noise_config = {}

        # Iterate through the flattened topology to aggregate required data columns and parsing rules
        for node_path, feature_configs in flat_squads.items():
            feature_list = [f.name for f in feature_configs]

            for f in feature_configs:
                if getattr(f, 'normalize', False):
                    features_to_normalize.append(f.name)
                
                # Extract and store the defined noise standard deviation to construct the perturbation map later
                noise_val = getattr(f, 'gaussian_noise_std', 0.0)
                if noise_val > 0.0:
                    feature_noise_config[f.name] = noise_val

            logger.info(f"Fetching data for node: {node_path} ({len(feature_list)} indicators requested)")

            res = self.fetcher.fetch_raw_data(symbol, timeframe, after_ms, until_ms, feature_list)

            if not res:
                continue

            for key, data_array in res.items():
                merged_features[key] = data_array
                squad_actual_keys[node_path].append(key)

        logger.info(f"Fetching dynamic target label: {target_feature}")
        raw_labels = self.fetcher.fetch_raw_data(symbol, timeframe, after_ms, until_ms, [target_feature])

        ohlcv_res = {}
        if include_ohlcv:
            ohlcv_keys = ["time_ms", "time", "open", "high", "low", "close", "volume"]
            raw_ohlcv = self.fetcher.fetch_raw_data(symbol, timeframe, after_ms, until_ms, ohlcv_keys)

            if raw_ohlcv:
                ohlcv_res = raw_ohlcv

        # Pre-flight data validation to prevent cryptic downstream matrix errors
        missing_reasons = []
        
        if not merged_features:
            missing_reasons.append("merged_features dictionary is empty (Failed to fetch or parse squad features).")

        if not raw_labels or target_feature not in raw_labels:
            missing_reasons.append(f"raw_labels dictionary is empty or missing '{target_feature}'.")

        if missing_reasons:
            logger.error("--- DATA VALIDATION FAILED ---")
            for reason in missing_reasons:
                logger.error(f"Reason: {reason}")
            logger.error("------------------------------")
            # Return empty constructs matching the expected return signature to safely abort the pipeline
            return ({}, torch.empty(0), None, None) if include_ohlcv else ({}, torch.empty(0), None)

        unified_labels = np.array(raw_labels[target_feature])
        synthetic_label_dict = {'dynamic_target': unified_labels.tolist()}

        sorted_feature_keys = sorted(list(merged_features.keys()))
        self.last_squad_map = squad_actual_keys

        # Delegate the sliding-window transformation and Z-score normalization to the processor
        X_unified, Y = self.processor.create_windowed_payload(
            features_dict=merged_features,
            labels_dict=synthetic_label_dict,
            feature_fields=sorted_feature_keys,
            label_field='dynamic_target',
            seq_len=config.SEQ_LEN,
            pre_pivot_fill=getattr(config, 'PRE_PIVOT_FILL', 0),
            post_pivot_fill=getattr(config, 'POST_PIVOT_FILL', 0),
            features_to_normalize=features_to_normalize
        )

        if X_unified.nelement() == 0:
            return ({}, torch.empty(0), None, None) if include_ohlcv else ({}, torch.empty(0), None)

        squad_tensors = {}
        squad_noise_maps = {}

        # Deconstruct the massive unified tensor back into independent, node-specific feature tensors
        for node_path, actual_api_keys in squad_actual_keys.items():
            if not actual_api_keys:
                continue

            indices = [sorted_feature_keys.index(k) for k in actual_api_keys]
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=X_unified.device)

            selected_features = torch.index_select(X_unified, 2, idx_tensor)
            squad_tensors[node_path] = selected_features.transpose(1, 2).contiguous()
            
            # Construct a dedicated noise standard deviation tensor for this specific node
            noise_values = [feature_noise_config.get(k, 0.0) for k in actual_api_keys]
            if any(v > 0.0 for v in noise_values):
                # Reshape to [Features, 1] to guarantee correct broadcasting against the sequence length dimension later
                noise_tensor = torch.tensor(noise_values, dtype=torch.float32, device=X_unified.device).view(-1, 1)
                squad_noise_maps[node_path] = noise_tensor

        if not include_ohlcv:
            return squad_tensors, Y, squad_noise_maps

        feat_matrix, _ = MarketDataProcessor.columnar_to_tensor(merged_features, sorted_feature_keys)
        label_matrix, _ = MarketDataProcessor.columnar_to_tensor(synthetic_label_dict, ['dynamic_target'])

        min_len = min(len(feat_matrix), len(label_matrix))

        aligned_ohlcv = {}

        # Truncate and shift the raw OHLCV arrays to perfectly align visually with the windowed tensor output
        if ohlcv_res:
            for key, values in ohlcv_res.items():
                arr = np.array(values)
                aligned_ohlcv[key] = arr[:min_len][config.SEQ_LEN - 1:]

        if "time" in aligned_ohlcv and "time_ms" not in aligned_ohlcv:
            aligned_ohlcv["time_ms"] = aligned_ohlcv["time"]

        return squad_tensors, Y, squad_noise_maps, aligned_ohlcv

    @staticmethod
    def build_dataloader(
        squad_tensors: dict[str, torch.Tensor],
        Y: torch.Tensor,
        batch_size: int,
        is_training: bool = False,
        noise_maps: dict[str, torch.Tensor] = None
    ):
        """
        Factory method to safely instantiate a FastDictDataLoader after ensuring tensor validity.

        Args:
            squad_tensors (dict[str, torch.Tensor]): The dictionary of parsed node tensors.
            Y (torch.Tensor): The parsed target label tensor.
            batch_size (int): Required batch size.
            is_training (bool, optional): Sets iteration mode. Defaults to False.
            noise_maps (dict[str, torch.Tensor], optional): Defines injection noise profiles. Defaults to None.

        Returns:
            FastDictDataLoader: The configured dataloader, or None if validation fails.
        """
        # Halt instantiation if the upstream pipeline failed to generate valid tensors
        if Y.nelement() == 0 or not squad_tensors:
            logger.error("Cannot build dataloader: Empty tensors provided.")
            return None

        return FastDictDataLoader(squad_tensors, Y, batch_size, is_training, noise_map=noise_maps)