import logging
from services.fed_learning import FED_CONFIG, VERBOSE_KEY

def logInfo(message: str):
    if(FED_CONFIG[VERBOSE_KEY]):
        logging.info(message)