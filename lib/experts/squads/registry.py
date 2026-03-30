class SquadRegistry:
    """Registry for managing available Squad classes.

    This allows centralized registration and lookup of different squad
    implementations, enabling dynamic loading and instantiation of squads
    by name.
    """
    # Internal dictionary mapping squad names to squad classes
    _squads = {}

    @classmethod
    def register(cls, name: str):
        """Register a Squad class under a given name.

        Intended to be used as a decorator on squad classes.

        Args:
            name (str): Unique identifier for the squad.

        Returns:
            Callable: Decorator that registers the class in the registry.
        """
        def decorator(squad_class):
            # Store the squad class under the provided name
            cls._squads[name] = squad_class
            return squad_class

        return decorator