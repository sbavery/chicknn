import re
import logging
from logging.handlers import RotatingFileHandler
import chicknn.config as config

logger = None

def setup_logger(console_level=None, file_level=None, filename=None):
    handlers = []
    logger = logging.getLogger(__name__)

    formatter = logging.Formatter('%(asctime)s | %(name)25s[%(lineno)4d] | %(levelname)8s | %(message)s', 
                                  datefmt="%Y-%m-%dT%H%M%S")

    if console_level:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if file_level and filename:
        file_handler = RotatingFileHandler(filename,
                            maxBytes=config.log_max_bytes,
                            backupCount=config.log_backups)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=console_level, handlers=handlers)

    return logger