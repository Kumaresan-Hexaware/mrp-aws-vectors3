import logging
from pathlib import Path

_INITIALIZED = False

def init_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    _INITIALIZED = True

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
