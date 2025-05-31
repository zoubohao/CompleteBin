import logging


def get_logger(name: str = "DeepMetaBin"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[ %(asctime)s ] - %(message)s')
    if not logger.handlers:
        console_hdr = logging.StreamHandler()
        console_hdr.setFormatter(formatter)
        logger.addHandler(console_hdr)
    return logger
