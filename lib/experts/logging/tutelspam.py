import logging
from lib.experts.logging.registry import LoggingRegistry


@LoggingRegistry.register('tutel_spam_filter')
class TutelSpamFilter(logging.Filter):
    """Custom logging filter to suppress specific Tutel logger spam messages.

    This filter detects Tutel-specific capacity messages and prevents them
    from being emitted by the logger, reducing noise in logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether a log record should be emitted.

        Args:
            record (logging.LogRecord): The log record to evaluate.

        Returns:
            bool: False if the record matches known Tutel spam patterns,
                  True otherwise.
        """
        msg = record.getMessage()

        # Suppress messages that contain Tutel capacity spam patterns
        if "Capacity =" in msg and "real-time capacity-factor" in msg:
            return False

        # Allow all other messages
        return True