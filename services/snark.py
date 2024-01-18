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
    try:
        response = requests.post(BASE_SNARK_SERVICE + '/prove', json={
            'input': {
                'id': id,
                'num': num
            }
        })
        json_response = response.json()
        proof, public_signals = json_response['proof'], json_response['publicSignals']
    except:
        proof, public_signals = {}, []

    return proof, public_signals


def zk_snark_verify(proof, public_signals) -> bool:
    try:
        response = requests.post(BASE_SNARK_SERVICE + '/verify', json={
            'proof': proof,
            'publicSignals': public_signals
        })
        return response.json()['verification']
    except:
        return False