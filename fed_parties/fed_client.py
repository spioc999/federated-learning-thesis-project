import numpy as np
from typing import Tuple, List
import tenseal as ts
import tenseal.enc_context as ts_enc
from services.ckks_he import create_ckks_encrypted_tensor_list, decrypt_tensors_and_scale
from services.keras_and_datasets import get_model

class FedClient:
    def __init__(
        self,
        id: int,
        train_dataset: Tuple,
        test_dataset: Tuple,
        context_ckks: ts.Context,
        secret_key: ts_enc.SecretKey,
    ):
        self.id = id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.context_ckks = context_ckks
        self.secret_key = secret_key


    def initialize_model(self, model_weights: list[np.array]):
        self.model = get_model()
        self.model.set_weights(model_weights)


    def fit(self, batch_size: int = 128, epochs: int = 2) -> List[ts.CKKSTensor]:
        x, y = self.train_dataset
        self.model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
        )

        enc_weights = create_ckks_encrypted_tensor_list(self.model.get_weights(), self.context_ckks)
        return enc_weights
    
    
    def update_model_with_encrypted_summed_weights(self, encrypted_weights: List[ts.CKKSTensor], num_sum: int) -> None:
        updated_weights = decrypt_tensors_and_scale(encrypted_weights, self.secret_key, num_sum)
        self.model.set_weights(updated_weights)


    def evaluate(self) -> Tuple[float, float]:
        x, y = self.test_dataset
        loss, accuracy = self.model.evaluate(x, y)
        return float(loss), float(accuracy)
