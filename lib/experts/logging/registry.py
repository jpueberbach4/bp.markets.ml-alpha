import logging

# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,  # Default logging level
    format='%(asctime)s | %(levelname)-7s | %(message)s'  # Timestamp, level, message
)

class LoggingRegistry:
    """Registry for managing custom logging filters.

    Allows centralized registration and later attachment of filters
    to any logging handler in the application.
    """
    # Internal dictionary mapping filter names to filter classes
    _loggers = {}

    @classmethod
    def register(cls, name: str):
        """Register a logging filter class under a given name.

        This is intended to be used as a decorator on filter classes.

        Args:
            name (str): Unique name for the logging filter.

        Returns:
            Callable: Decorator that registers the class in the registry.
        """
        def decorator(filter_class):
            # Store the filter class under the provided name
            cls._loggers[name] = filter_class
            return filter_class

        return decorator


def register_logging_filters():
    """Attach all registered logging filters to existing root handlers.

    Iterates over all handlers of the root logger and instantiates
    each filter class in the LoggingRegistry, adding it to the handler.
    """
    # Iterate over each root logger handler
    for handler in logging.root.handlers:
        # Attach all registered filters
        for filter_class in LoggingRegistry._loggers.values():
            handler.addFilter(filter_class())


import lib.experts.logging.tutelspam # Register the Tutel log-spam killer
register_logging_filters()