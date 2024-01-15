import numpy as np
from typing import Tuple, List
import tenseal as ts
import tenseal.enc_context as ts_enc
from services.ckks_he import create_ckks_encrypted_tensor_list, decrypt_tensors
from services.keras_and_datasets import get_model

class FedClient:
    def __init__(
        self,
        id: int,
        train_dataset: Tuple,
        test_dataset: Tuple,
        context_ckks: ts.Context=None,
        secret_key: ts_enc.SecretKey=None,
    ):
        self.id = id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.context_ckks = context_ckks
        self.secret_key = secret_key


    def initialize_model(self, model_weights: List[np.array]) -> None:
        self.model = get_model()
        self.set_model_weights(model_weights)


    def fit(self, batch_size: int = 128, epochs: int = 2) -> List[np.array]:
        x, y = self.train_dataset
        self.model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
        )
        return self.get_model_weights()


    def fit_with_he(self, batch_size: int = 128, epochs: int = 2) -> List[ts.CKKSTensor]:
        weigths = self.fit(batch_size=batch_size, epochs=epochs)
        enc_weights = create_ckks_encrypted_tensor_list(weigths, self.context_ckks)
        return enc_weights
    

    def get_model_weights(self) -> List[np.array]:
        return self.model.get_weights()
    

    def set_model_weights(self, weigths: List[np.array]) -> None:
        self.model.set_weights(weigths)


    def update_model(self, weigths: List[np.array], scale: int) -> None:
        scaled_weights = [arrays / scale for arrays in weigths]
        self.set_model_weights(scaled_weights)
    
    
    def update_model_with_he(self, encrypted_weights: List[ts.CKKSTensor], scale: int) -> None:
        updated_weights = decrypt_tensors(encrypted_weights, self.secret_key)
        self.update_model(updated_weights, scale)


    def evaluate(self) -> Tuple[float, float]:
        x, y = self.test_dataset
        loss, accuracy = self.model.evaluate(x, y)
        return float(loss), float(accuracy)
    

    def evaluate_with_zk_snark(self):
        pass #TODO complete me 
