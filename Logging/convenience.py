import logging

from typing import Optional


_NULL_LOGGER = logging.Logger("data_utilities.null", level=logging.CRITICAL)


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Return the supplied logger or a logger that discards debug messages.

    Arguments
    ---------
    logger: Optional[logging.Logger] (default None)
        Optional logger to use.

    Returns
    -------
        The supplied logger or the null logger.
    """

    return logger if logger is not None else _NULL_LOGGER
