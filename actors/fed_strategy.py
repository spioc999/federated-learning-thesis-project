from actors.fed_client import FedClient
from services.keras_and_datasets import get_model
import random
import threading

class FedStrategy:
    def __init__(
        self,
        clients: list[FedClient],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
    ):
        self.clients = clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate

    
    def initialize_models(self):
        """Initialize global model parameters."""
        init_model_weights = get_model().get_weights()
        
        for client in self.clients:
            client.initialize_model(init_model_weights)


    def run_distributed_fit(self, fit_round: int) -> None:
        num_fit_clients = round(len(self.clients) * self.fraction_fit)
        fit_clients = random.sample(self.clients, num_fit_clients)

