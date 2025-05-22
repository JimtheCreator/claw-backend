import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def configure_logging(name: str = "ClawBackend") -> logging.Logger:
    """Configure and return a logger with file and stream handlers"""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        encoding='utf-8',
        force=True
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.encoding = 'utf-8'

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Save original error method
    original_error = logger.error

    # Add exception hook
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = exception_handler

    # Custom error logger
    def error_with_traceback(msg, *args, **kwargs):
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            tb = traceback.extract_tb(exc_info[2])
            filename, line_number, func, text = tb[-1]
            error_location = f"{filename}:{line_number} in {func}"
            exc_type = exc_info[0].__name__
            exc_msg = str(exc_info[1])
            detailed_msg = f"{msg} | Exception: {exc_type}: {exc_msg} at {error_location}"
            if text:
                detailed_msg += f" | Code: {text}"

            # Avoid double exc_info by removing it from kwargs if present
            kwargs.pop("exc_info", None)

            original_error(detailed_msg, *args, exc_info=exc_info, **kwargs)
        else:
            original_error(msg, *args, **kwargs)

    logger.error = error_with_traceback

    return logger

# Init logger
logger = configure_logging()
