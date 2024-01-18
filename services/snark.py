from typing import Tuple
import requests
import sys
from services.logger import log_info


BASE_SNARK_SERVICE = 'http://localhost:3000'

def check_zk_snark() -> Tuple:
    try:
        response = requests.get(BASE_SNARK_SERVICE + '/health')
        if not response.json()['status'] == 'ok':
            raise Exception('Not ok')
    except:
        log_info('[MAIN] ERROR_SETUP | ZK not available')
        sys.exit()



def zk_snark_prove(id: int, num: int) -> Tuple:
    proof, publicSignals = None, None
    return proof, publicSignals


def zk_snark_verify(proof, publicSignals) -> bool:
    return True