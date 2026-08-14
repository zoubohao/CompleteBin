import logging
import sys

def get_logger(name: str = "CompleteBin"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(name)-16s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    if not logger.handlers:
        console_hdr = logging.StreamHandler(sys.stdout)
        console_hdr.setFormatter(formatter)
        logger.addHandler(console_hdr)
    return logger
