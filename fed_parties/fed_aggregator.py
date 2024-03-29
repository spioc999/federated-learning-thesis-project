from fed_parties.fed_client import FedClient
from services.keras_and_datasets import get_model
import random
import numpy as np
from typing import List, Dict
from multiprocessing.pool import ThreadPool
import sys
from services.snark import zk_snark_verify

HE_CONFIG_KEY = 'he'
ZK_CONFIG_KEY = 'zk'

class FedAggregator:
    def __init__(
        self,
        clients: List[FedClient],
        config: Dict[str, bool],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.3,
    ):
        self.clients = clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.config = config
        self.history = {
            'fit': [],
            'evaluate': []
        }
        self.model = get_model()



    
    def _aggregator_log(self, message: str):
        from services.logger import log_info
        log_info(f'[FedAggregator] {message}')
    



    def initialize(self) -> None:
        self._aggregator_log('INITIALIZE | Started')

        self._aggregator_log(f'INITIALIZE | CONFIG | Clients: {len(self.clients)} - Fraction_fit: {self.fraction_fit} - Fraction_evaluate: {self.fraction_evaluate}')
        
        init_model_weights = self.model.get_weights()
        with ThreadPool() as pool:
            pool.map(
                lambda client: client.initialize_model(init_model_weights),
                self.clients,
            )

        self._aggregator_log('INITIALIZE | Completed')




    def run_distributed_fit(self, fed_round: int) -> None:
        self._aggregator_log(f'FIT | ROUND {fed_round} | Started')

        num_fit_clients = round(len(self.clients) * self.fraction_fit)
        fit_clients = random.sample(self.clients, num_fit_clients)
        self._aggregator_log(f'FIT | ROUND {fed_round} | Sampled {len(fit_clients)} clients')
        
        self._aggregator_log(f'FIT | ROUND {fed_round} | CLIENTS_FIT | Started')
        clients_weights = []
        with ThreadPool() as pool:
            for fit_result in pool.map(
                lambda client: 
                    client.run_fit(fed_round, self.config[HE_CONFIG_KEY], self.config[ZK_CONFIG_KEY]),
                fit_clients
            ):
                if self.config[ZK_CONFIG_KEY]:
                    fit_weights, proof, public_signals, client_index = fit_result
                    if zk_snark_verify(proof, public_signals):
                        self._aggregator_log(f'FIT | ROUND {fed_round} | CLIENTS_FIT | ZK [OK] FedClient#{client_index}')
                        clients_weights.append(fit_weights)  
                    else:
                        self._aggregator_log(f'FIT | ROUND {fed_round} | CLIENTS_FIT | ZK [NO] FedClient#{client_index}')
                else:
                    clients_weights.append(fit_result)   

        self._aggregator_log(f'FIT | ROUND {fed_round} | CLIENTS_FIT | Completed - Size clients_weights: {sys.getsizeof(clients_weights)}B')

        self._aggregator_log(f'FIT | ROUND {fed_round} | AGGREGATION_WEIGHTS | Started')
        summed_weights = [client_weights for client_weights in clients_weights[0]]
        for client_weights in clients_weights[1:]:
            for i, weights in enumerate(client_weights):
                summed_weights[i] = summed_weights[i] + weights
        num_summ = len(summed_weights)
        self._aggregator_log(f'FIT | ROUND {fed_round} | AGGREGATION_WEIGHTS | Completed')

        self._aggregator_log(f'FIT | ROUND {fed_round} | UPDATING_CLIENTS_WITH_AGGREGATED_WEIGHTS | Started')
        with ThreadPool() as pool:
            pool.map(
                lambda client: 
                    client.scale_weights_and_update_model_with_he(summed_weights, num_summ) if self.config[HE_CONFIG_KEY]
                    else client.scale_weights_and_update_model(summed_weights, num_summ),
                self.clients,
            )
        self._aggregator_log(f'FIT | ROUND {fed_round} | UPDATING_CLIENTS_WITH_AGGREGATED_WEIGHTS | Completed')

        self._aggregator_log(f'FIT | ROUND {fed_round} | Completed')




    def run_get_aggregated_model_and_align_with_voting(self, fed_round: int) -> None:
        self._aggregator_log(f'GET_AGGREGATED_MODEL | ROUND {fed_round} | Started')

        clients_weights = []
        with ThreadPool() as pool:
            if self.config[ZK_CONFIG_KEY]:
                for client_result in pool.map(lambda client: client.get_model_weights_with_snark(), self.clients):
                    client_weights, client_proof, client_public_signals, client_index = client_result
                    self._aggregator_log(f'GET_AGGREGATED_MODEL | ROUND {fed_round} | ZK verifying FedClient#{client_index} ...')
                    if zk_snark_verify(client_proof, client_public_signals):
                        self._aggregator_log(f'GET_AGGREGATED_MODEL | ROUND {fed_round} | ZK [OK] FedClient#{client_index}')
                        clients_weights.append(client_weights)
                    else:
                        self._aggregator_log(f'GET_AGGREGATED_MODEL | ROUND {fed_round} | ZK [NO] FedClient#{client_index}')
            else:
                for client_weights in pool.map(lambda client: client.get_model_weights(), self.clients):
                    clients_weights.append(client_weights)


        self._aggregator_log(f'GET_AGGREGATED_MODEL | ROUND {fed_round} | MAX_VOTING weights')
        weights = FedAggregator._get_most_voted_weights(clients_weights)

        self._aggregator_log(f'GET_AGGREGATED_MODEL | ROUND {fed_round} | Aligning all models and aggregator model')
        with ThreadPool() as pool:
            pool.map(
                lambda client: client.set_model_weights(weights),
                self.clients,
            )
        self.model.set_weights(weights) #Update aggregator model
        
        self._aggregator_log(f'GET_AGGREGATED_MODEL | ROUND {fed_round} | Completed')


    def _get_most_voted_weights(weights: List) -> None:
        voting_dict = {}
        hash_dict = {}
        for w in weights:
            hash_w = hash(str(w))
            hash_dict[hash_w] = w
            if hash_w in voting_dict:
                voting_dict[hash_w] = voting_dict[hash_w] + 1
            else:
                voting_dict[hash_w] = 1

        most_voted = max(voting_dict, key=lambda k: voting_dict[k])
        return hash_dict[most_voted]


    def run_distributed_evaluate(self, fed_round: int) -> None:
        self._aggregator_log(f'EVALUATE | ROUND {fed_round} | Started')

        num_eval_clients = round(len(self.clients) * self.fraction_evaluate)
        eval_clients = random.sample(self.clients, num_eval_clients)
        self._aggregator_log(f'EVALUATE | ROUND {fed_round} | Sampled {len(eval_clients)} clients')

        losses = []
        accuracies = []
        
        self._aggregator_log(f'EVALUATE | ROUND {fed_round} | CLIENTS_EVALUATE | Started')
        with ThreadPool() as pool:
            for eval_result in pool.map(
                    lambda client:
                        client.evaluate(fed_round),
                    eval_clients
                ):
                loss, accuracy = eval_result
                losses.append(loss)
                accuracies.append(accuracy)
        self._aggregator_log(f'EVALUATE | ROUND {fed_round} | CLIENTS_EVALUATE | Completed')

        mean_loss, mean_accuracy = np.mean(losses), np.mean(accuracies)
        self.history['evaluate'].append((fed_round, mean_loss, mean_accuracy))
        
        self._aggregator_log(f'EVALUATE | ROUND {fed_round} | AGGREGATED_METRICS | Loss: {mean_loss} - Accuracy: {mean_accuracy}')
        self._aggregator_log(f'EVALUATE | ROUND {fed_round} | Completed')
