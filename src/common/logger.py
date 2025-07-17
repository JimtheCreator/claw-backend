import logging
import sys
import traceback
# Import the process-safe handler
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pathlib import Path

# Create a directory for log files if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def configure_logging(name: str = "ClawBackend") -> logging.Logger:
    """
    Configure and return a logger with process-safe file rotation and stream handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # 1. Use ConcurrentRotatingFileHandler to prevent file locking errors
    # This handler is process-safe and uses a lock file to coordinate rotation.
    file_handler = ConcurrentRotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 2. Console handler that writes to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Set a global hook for uncaught exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = exception_handler
    
    # Keep the logger's propaganda to a minimum to avoid cluttering other modules
    logger.propagate = False

    return logger

# Initialize and export the logger for use in other modules
logger = configure_logging()
