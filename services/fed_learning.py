from fed_parties.fed_aggregator import FedAggregator
from fed_parties.fed_client import FedClient
from typing import List
from services.keras_and_datasets import load_datasets
from multiprocessing.pool import ThreadPool
from services.ckks_he import generate_context_and_secret
from services.logger import log_info, setup_logger
import datetime
import random
import tensorflow as tf

HE_CONFIG_KEY = 'he'
ZK_CONFIG_KEY = 'zk'

FED_CONFIG = {
    HE_CONFIG_KEY: False,
    ZK_CONFIG_KEY: False,
}

SEED = 42

def _generate_clients(num_clients: int) -> List[FedClient]:
    x_train_datasets, y_train_datasets, x_test_datasets, y_test_datasets = load_datasets(num_clients)
    return [
        FedClient(
            id=i,
            train_dataset=(x_train_datasets[i], y_train_datasets[i]),
            test_dataset=(x_test_datasets[i], y_test_datasets[i]),
        )
        for i in range(num_clients)
    ]


def _setup_config(enable_he: bool, enable_zk_proof: bool, verbose: bool):
    global FED_CONFIG
    FED_CONFIG[HE_CONFIG_KEY] = enable_he
    FED_CONFIG[ZK_CONFIG_KEY] = enable_zk_proof
    setup_logger(verbose=verbose)
    random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    
def start_fed_averaging_mnist_simulation(num_clients: int, num_rounds: int, fraction_fit: float = 1.0,
                              fraction_evaluate: float = 0.3, enable_he: bool = False,
                              enable_zk_proof: bool = False, verbose: bool = True) -> None:
    
    start_datetime = datetime.datetime.now()
    _setup_config(enable_he, enable_zk_proof, verbose)
    log_info(f'[MAIN] Starting FedAveraging MNIST simulation...')
    log_info(f'[MAIN] SETUP | Fed_Config: {FED_CONFIG} - Verbose: {verbose}')

    if(FED_CONFIG[ZK_CONFIG_KEY]):
        pass #TODO


    log_info(f'[MAIN] SETUP | FedClients created!')

    aggregator = FedAggregator(
        clients=_generate_clients(num_clients),
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        config= FED_CONFIG
    )
    log_info(f'[MAIN] SETUP | FedAggregator created and ready!')

    aggregator.initialize()

    for round in range(1, num_rounds + 1):
        log_info(f'[MAIN] ROUND {round} | Started')
        start_round_datetime = datetime.datetime.now()

        if(FED_CONFIG[HE_CONFIG_KEY]):
            log_info(f'[MAIN] ROUND {round} | Creating homomorphic encryption keys and set in clients...')
            context_ckks, secret_key = generate_context_and_secret()
            with ThreadPool() as pool:
                pool.map(
                    lambda client: client.set_ckks_context_and_secret_key(context_ckks, secret_key),
                    aggregator.clients,
                )
            log_info(f'[MAIN] ROUND {round} | Homomorphic encryption keys created!')

        aggregator.run_distributed_fit(fed_round=round)
        aggregator.run_get_aggregated_model_and_align_clients(fed_round=round)
        aggregator.run_distributed_evaluate(fed_round=round)
        
        end_round_datetime = datetime.datetime.now()
        log_info(f'[MAIN] ROUND {round} | Completed in {end_round_datetime - start_round_datetime}')

    log_info(f'[MAIN] Evaluation metrics history (round, loss, accuracy):\n{aggregator.history["evaluate"]}')
    end_datetime = datetime.datetime.now()
    log_info(f'[MAIN] Completed FedAveraging MNIST simulation successfully in {end_datetime - start_datetime}!')