import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Named logger instance for this module
logger = logging.getLogger("DataFetcher")

class BaseMarketDataFetcher(ABC):
    """Abstract base class for fetching and transforming market data.

    Provides a standardized interface for subclasses to implement
    data retrieval from external APIs and conversion into model-ready tensors.
    """

    def __init__(self, base_url: str, timeout: int = 10):
        """Initialize the fetcher with API connection parameters.

        Stores the base API URL and request timeout for use in
        concrete data fetchers.

        Args:
            base_url (str): Base URL of the API endpoint.
            timeout (int, optional): Request timeout in seconds. Defaults to 10.
        """
        # Base URL of the API
        self.base_url = base_url

        # Timeout in seconds for API requests
        self.timeout = timeout

    @abstractmethod
    def fetch_raw_data(
        self,
        symbol: str,
        timeframe: str,
        after_ms: int,
        until_ms: int,
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """Retrieve raw market data from an external source.

        Must be implemented by subclasses to handle API-specific
        request logic, including authentication and response parsing.

        Args:
            symbol (str): Market symbol (e.g., "BTC-USD").
            timeframe (str): Data timeframe (e.g., "1m", "1h").
            after_ms (int): Start timestamp in milliseconds.
            until_ms (int): End timestamp in milliseconds.
            fields (List[str]): List of data fields to retrieve.

        Returns:
            List[Dict[str, Any]]: Raw market data as a list of dictionaries,
            where each dictionary represents one data record.
        """
        # Abstract method: implementation required in subclass
        pass

    @abstractmethod
    def transform_to_tensors(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw API response data into tensors for models.

        Subclasses must implement this method to transform dictionary-based
        API responses into numerical tensors compatible with sequence models.

        Args:
            raw_data (List[Dict[str, Any]]): Raw data returned by fetch_raw_data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - features: Tensor of shape [batch, seq_len, indicator_dim].
                - labels: Tensor of shape [batch, 1], aligned with features.
        """
        # Abstract method: implementation required in subclass
        pass