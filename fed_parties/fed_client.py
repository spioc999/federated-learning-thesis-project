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
    ):
        self.id = id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = get_model()


    def set_ckks_context_and_secret_key(self, context_ckks: ts.Context, secret_key: ts_enc.SecretKey) -> None:
        self._client_log(f'SET_CKKS | Setting CKKS context and secret key')
        self.context_ckks = context_ckks
        self.secret_key = secret_key


    def _client_log(self, message: str):
        from services.logger import log_info
        log_info(f'[FedClient#{self.id}] {message}')




    def initialize_model(self, model_weights: List[np.array]) -> None:
        self._client_log(f'INITIALIZE | Initializing model weights')
        self.set_model_weights(model_weights, _force_print_logs=False)




    def fit(self, round: int, batch_size: int = 128, epochs: int = 2, _force_print_logs: bool = True) -> List[np.array]:
        if _force_print_logs: self._client_log(f'FIT | ROUND {round} | Batch_Size: {batch_size} - Epochs: {epochs} | Started')
        x, y = self.train_dataset
        self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )
        if _force_print_logs: self._client_log(f'FIT | ROUND {round} | Completed and returning weights')
        return self.get_model_weights()




    def fit_with_he(self, round: int, batch_size: int = 128, epochs: int = 2) -> List[ts.CKKSTensor]:
        self._client_log(f'FIT_WITH_HE | ROUND {round} | Batch_Size: {batch_size} - Epochs: {epochs} | Started')
        weigths = self.fit(round=round, batch_size=batch_size, epochs=epochs, _force_print_logs=False)
        self._client_log(f'FIT_WITH_HE | ROUND {round} | Fit completed and encrypting weights')
        enc_weights = create_ckks_encrypted_tensor_list(weigths, self.context_ckks)
        self._client_log(f'FIT_WITH_HE | ROUND {round} | Completed and returning CKKS encrypted weights')
        return enc_weights
    



    def get_model_weights(self) -> List[np.array]:
        return self.model.get_weights()
    



    def set_model_weights(self, weigths: List[np.array], _force_print_logs: bool = True) -> None:
        if _force_print_logs: self._client_log(f'SET_MODEL_WEIGHTS | Setting weights')
        self.model.set_weights(weigths)




    def update_model(self, weigths: List[np.array], scale: int) -> None:
        self._client_log(f'UPDATE_MODEL | Scaling weights')
        scaled_weights = [arrays / scale for arrays in weigths]
        self.set_model_weights(scaled_weights)
    


    
    def update_model_with_he(self, encrypted_weights: List[ts.CKKSTensor], scale: int) -> None:
        self._client_log(f'UPDATE_MODEL_WITH_HE | Decrypting weights')
        updated_weights = decrypt_tensors(encrypted_weights, self.secret_key)
        self._client_log(f'UPDATE_MODEL_WITH_HE | Weights decrypted')
        self.update_model(updated_weights, scale)




    def evaluate(self, round: int, _force_print_logs: bool = True) -> Tuple[float, float]:
        if _force_print_logs: self._client_log(f'EVALUATE | ROUND {round} | Started')
        x, y = self.test_dataset
        loss, accuracy = self.model.evaluate(x, y, verbose=0)
        if _force_print_logs: self._client_log(f'EVALUATE | ROUND {round} | Completed')
        return float(loss), float(accuracy)
    

    

    def evaluate_with_zk_snark(self, round: int):
        self._client_log(f'EVALUATE_WITH_ZK_SNARK | ROUND {round} | Started')
        #TODO complete me 
        self._client_log(f'EVALUATE_WITH_ZK_SNARK | ROUND {round} | Completed')
