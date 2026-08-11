class BaseCallback:
    """Base class for lifecycle callbacks.

    Every callback receives keyword-only event context.  All events include
    ``experiment_id`` and ``seed``; each event-specific base class documents
    its additional values.
    """

    def __init__(self, logger=None, **kwargs) -> None:
        """
        initialize attributes common to all adtool.legacy callbacks

        Args:
            logger: logger used to report experiment progress.
        """
        self.logger = logger

    def __call__(self, **context) -> None:
        """Handle one lifecycle event using its keyword context."""
        raise NotImplementedError
