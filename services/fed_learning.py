from fed_parties.fed_aggregator import FedAggregator
from fed_parties.fed_client import FedClient
from typing import List
from services.keras_and_datasets import load_datasets
import tenseal as ts
import tenseal.enc_context as ts_enc
from ckks_he import generate_context_and_secret
from services.logger import logInfo

HE_CONFIG_KEY = 'homomorphic_encryption'
ZK_CONFIG_KEY = 'zero-knowledge_proof'
VERBOSE_KEY = 'verbose'

FED_CONFIG = {
    HE_CONFIG_KEY: False,
    ZK_CONFIG_KEY: False,
    VERBOSE_KEY: False
}


def _generate_clients(num_clients: int, context_ckks: ts.Context = None, secret_key: ts_enc.SecretKey = None) -> List[FedClient]:
    x_train_datasets, y_train_datasets, x_test_datasets, y_test_datasets = load_datasets(num_clients)
    return [
        FedClient(
            id=i,
            train_dataset=(x_train_datasets[i], y_train_datasets[i]),
            test_dataset=(x_test_datasets[i], y_test_datasets[i]),
            context_ckks=context_ckks,
            secret_key=secret_key,
        )
        for i in range(num_clients)
    ]


def _setup_config(enable_he: bool, enable_zk_proof: bool, verbose: bool):
    global FED_CONFIG
    FED_CONFIG[HE_CONFIG_KEY] = enable_he
    FED_CONFIG[ZK_CONFIG_KEY] = enable_zk_proof
    FED_CONFIG[VERBOSE_KEY] = verbose


def start_fed_averaging_mnist_simulation(num_clients: int, num_rounds: int, fraction_fit: float = 1.0,
                              fraction_evaluate: float = 0.3, enable_he: bool = False,
                              enable_zk_proof: bool = False, verbose: bool = False) -> None:
    
    logInfo(f'[MAIN] Starting FedAveraging MNIST simulation...')
    _setup_config(enable_he, enable_zk_proof, verbose)
    logInfo(f'[MAIN] Simulation config: {FED_CONFIG}')

    context_ckks, secret_key = None, None

    if(FED_CONFIG[HE_CONFIG_KEY]):
        logInfo(f'[MAIN] Creating homomorphic encryption keys...')
        context_ckks, secret_key = generate_context_and_secret()
        logInfo(f'[MAIN] Homomorphic encryption keys created!')


    clients = _generate_clients(num_clients, context_ckks, secret_key)

    logInfo(f'[MAIN] FedClients created!')

    aggregator = FedAggregator(
        clients=clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
    )

    logInfo(f'[MAIN] FedAggregator created and ready!')

    aggregator.initialize_models()

    for round in range(num_rounds):
        logInfo(f'\n[MAIN] Started ROUND {round + 1}')

        