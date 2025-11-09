import logging
import sys

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def setup_logging(log_level: str):
    level = _LOG_LEVEL_MAP.get(log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s %(levelname)8s [%(name)s]: %(message)s")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root_logger.addHandler(handler)

    

def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)
