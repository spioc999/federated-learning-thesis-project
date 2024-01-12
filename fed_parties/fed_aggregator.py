from fed_parties.fed_client import FedClient
from services.keras_and_datasets import get_model
import random
from multiprocessing.pool import ThreadPool
import numpy as np

class FedAggregator:
    def __init__(
        self,
        clients: list[FedClient],
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

    
    def initialize_models(self):
        """Initialize global model parameters."""
        init_model_weights = get_model().get_weights()

        with ThreadPool() as pool:
            pool.map(
                lambda client: client.initialize_model(init_model_weights),
                self.clients,
            )


    def run_distributed_fit(self, fed_round: int) -> None:
        num_fit_clients = round(len(self.clients) * self.fraction_fit)
        fit_clients = random.sample(self.clients, num_fit_clients)

        clients_enc_weights = []
        
        # Distribute fit
        with ThreadPool() as pool:
            for fit_result in pool.map(lambda client: client.fit(), fit_clients):
                clients_enc_weights.append(fit_result)

        # Homomorphic addition (aggregation)
        he_summed_weights = [enc_weights for enc_weights in clients_enc_weights[0]]

        for client_enc_weights in clients_enc_weights[1:]:
            for i, enc_weights in enumerate(client_enc_weights):
                he_summed_weights[i] = he_summed_weights[i] + enc_weights

        num_summ = len(clients_enc_weights)

        # Update all clients
        with ThreadPool() as pool:
            pool.map(
                lambda client: client.update_model_with_encrypted_summed_weights(he_summed_weights, num_summ),
                self.clients,
            )


    def run_distributed_evaluate(self, fed_round: int) -> None:
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
