import requests
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Logger instance specific to this module
logger = logging.getLogger("DataFetcher")

class BaseMarketDataFetcher(ABC):
    """Abstract base class for retrieving market data.

    Defines a standardized interface for all market data fetchers,
    ensuring downstream RPulsar Experts receive data in consistent format.
    """

    def __init__(self, base_url: str, timeout: int = 10):
        """Initialize the fetcher with API configuration.

        Args:
            base_url (str): Base URL of the REST API.
            timeout (int, optional): Request timeout in seconds. Defaults to 10.
        """
        # Base URL for API requests
        self.base_url = base_url

        # Timeout in seconds for HTTP requests
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

        Must be implemented by subclasses to handle API-specific request
        construction, authentication, and parsing.

        Args:
            symbol (str): Market symbol (e.g., "USD-JPY").
            timeframe (str): Data timeframe (e.g., "4h").
            after_ms (int): Start timestamp in milliseconds.
            until_ms (int): End timestamp in milliseconds.
            fields (List[str]): List of fields to fetch.

        Returns:
            List[Dict[str, Any]]: Raw data records, each as a dictionary.
        """
        # Abstract method placeholder
        pass


class RESTMarketFetcher(BaseMarketDataFetcher):
    """Concrete implementation for fetching OHLCV data via REST API 1.1.

    Supports:
        - Bracketed field selection (e.g., [hma(9):rsi(14)])
        - Path-based filtering using timestamps
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the REST fetcher with optional base URL.

        Args:
            base_url (str, optional): Base API endpoint. Defaults to localhost.
        """
        # Call parent constructor to store base URL and default timeout
        super().__init__(base_url)

    def fetch_raw_data(
        self,
        symbol: str,
        timeframe: str,
        after_ms: int,
        until_ms: int,
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch OHLCV data from the REST API with field and time filtering.

        Constructs the API request endpoint according to RPulsar OHLCV 1.1
        specification, executes the GET request, and filters the result
        to include only requested fields.

        Args:
            symbol (str): Market symbol (e.g., "USD-JPY").
            timeframe (str): Timeframe string (e.g., "4h").
            after_ms (int): Start timestamp in milliseconds.
            until_ms (int): End timestamp in milliseconds.
            fields (List[str]): List of requested field names.

        Returns:
            List[Dict[str, Any]]: Filtered result data if successful;
            returns empty list on failure.
        """
        # Convert fields list to API-specific bracketed string format
        fields_str = f"[{':'.join(fields)}]"

        # Construct endpoint path for OHLCV REST API
        endpoint_path = (
            f"/ohlcv/1.1/select/{symbol},{timeframe}{fields_str}"
            f"/after/{after_ms}/until/{until_ms}/output/JSON"
        )

        # Combine base URL and endpoint path
        full_url = f"{self.base_url.rstrip('/')}{endpoint_path}"

        # Static query parameters for API request
        params = {
            "limit": 1000000,  # Maximum number of records
            "subformat": 3,    # API-specific formatting option
            "order": "asc"     # Return data in ascending order
        }

        try:
            # Log the full request URL for transparency
            logger.info(f"Requesting URI: {full_url}")

            # Execute GET request with timeout
            response = requests.get(full_url, params=params, timeout=self.timeout)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Parse JSON response
            data = response.json()

            # Handle wrapped response in 'result' key
            if isinstance(data, dict) and 'result' in data:
                result = {}

                # Iterate over returned fields
                for k in data['result'].keys():
                    # Extract field name without suffix
                    field_name, _ = k.split('__', 1) if '__' in k else [k, None]

                    # Include only requested fields
                    if field_name in fields:
                        result[k] = data['result'][k]

                return result

            # Return None if response format is unexpected
            return None

        except requests.exceptions.RequestException as e:
            # Log request error with full URL
            logger.error(f"Failed to fetch data from {full_url}: {e}")

            # Return empty list to indicate failure
            return []