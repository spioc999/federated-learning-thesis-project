import logging

def setup_logger(verbose: bool=True):
    if verbose:
        logging.basicConfig(
            format='%(asctime)s.%(msecs)03d %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%dT%H:%M:%S'
        )

def log_info(message: str):
    logging.info(message)