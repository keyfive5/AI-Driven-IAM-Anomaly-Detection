import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up comprehensive logging configuration for the application.
    
    Args:
        log_dir (str): Directory where log files will be stored
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_path / f"iam_anomaly_detection_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (for all logs)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler (for INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers for different components
    loggers = {
        'data': logging.getLogger('data'),
        'model': logging.getLogger('model'),
        'gui': logging.getLogger('gui'),
        'analysis': logging.getLogger('analysis')
    }
    
    # Configure component-specific loggers
    for logger in loggers.values():
        logger.setLevel(logging.DEBUG)
        logger.propagate = True  # Propagate logs to root logger
    
    # Log initial setup
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log file: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name (str): Name of the component (e.g., 'data', 'model', 'gui', 'analysis')
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name) 