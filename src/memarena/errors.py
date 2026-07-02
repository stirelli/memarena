class ProviderError(Exception):
    """Raised by a MemoryProvider adapter on failure.

    The runner catches this and records the item as `infra_error`,
    excluded from accuracy and reported separately.
    """
