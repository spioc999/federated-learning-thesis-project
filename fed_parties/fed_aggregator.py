from fed_parties.fed_client import FedClient
from services.keras_and_datasets import get_model
import random
from multiprocessing.pool import ThreadPool
import numpy as np
from services.fed_learning import FED_CONFIG, HE_CONFIG_KEY, ZK_CONFIG_KEY
from typing import List

class FedAggregator:
    def __init__(
        self,
        clients: List[FedClient],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.3,
    ):
        self.clients = clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.history = {
            'fit': [],
            'evaluate': []
        }

    
    def _aggregator_log(self, message: str):
        from services.logger import logInfo
        logInfo(f'[FedAggregator] {message}')
    

    def initialize_models(self) -> None:
        """Initialize global model parameters."""

        self._aggregator_log('Initializing models...')
        init_model_weights = get_model().get_weights()

        with ThreadPool() as pool:
            pool.map(
                lambda client: client.initialize_model(init_model_weights),
                self.clients,
            )

        self._aggregator_log('Initialization of models COMPLETED!')


    def run_distributed_fit(self, fed_round: int) -> None:
        self._aggregator_log(f'Starting FIT - ROUND {fed_round}')

        num_fit_clients = round(len(self.clients) * self.fraction_fit)
        fit_clients = random.sample(self.clients, num_fit_clients)

        clients_weights = []
        
        # Distribute fit
        with ThreadPool() as pool:
            for fit_result in pool.map(
                lambda client: 
                    client.fit_with_he() if FED_CONFIG[HE_CONFIG_KEY]
                    else client.fit(),
                fit_clients
            ):
                clients_weights.append(fit_result)

        # Addition (aggregation)
        summed_weights = [client_weights for client_weights in clients_weights[0]]

        for client_enc_weights in clients_weights[1:]:
            for i, enc_weights in enumerate(client_enc_weights):
                summed_weights[i] = summed_weights[i] + enc_weights

        num_summ = len(clients_weights)

        # Update all clients
        with ThreadPool() as pool:
            pool.map(
                lambda client: 
                    client.update_model_with_he(summed_weights, num_summ) if FED_CONFIG[HE_CONFIG_KEY]
                    else client.update_model(summed_weights, num_summ),
                self.clients,
            )


    def run_check_models(self, fed_round: int) -> None:
        clients_weights = []
        with ThreadPool() as pool:
            for client_weights in pool.map(
                lambda client: client.get_model_weights(),
                self.clients
            ):
                clients_weights.append(client_weights)

        return all(i == clients_weights[0] for i in clients_weights)


    def run_distributed_evaluate(self, fed_round: int) -> None:
        #TODO complete me
        num_eval_clients = round(len(self.clients) * self.fraction_evaluate)
        eval_clients = random.sample(self.clients, num_eval_clients)

        losses = []
        accuracies = []
        
        # Distribute evaluate
        with ThreadPool() as pool:
            for eval_result in pool.map(lambda client: client.evaluate(), eval_clients):
                loss, accuracy = eval_result
                losses.append(loss)
                accuracies.append(accuracy)

        self.history['evaluate'].append((fed_round, np.mean(losses), np.mean(accuracies)))
