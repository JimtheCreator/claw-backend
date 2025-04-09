import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def configure_logging(name: str = "ClawBackend") -> logging.Logger:
    """Configure and return a logger with file and stream handlers"""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='utf-8',  # Add this line
        force=True
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)


    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.encoding = 'utf-8'  # Add this line

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Configure root logger
logger = configure_logging()