import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup production-grade logger"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
   
    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
