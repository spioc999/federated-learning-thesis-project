import tenseal as ts
import tenseal.enc_context as ts_enc_context
import numpy as np
from typing import Tuple, List
from memory_profiler import profile

@profile
def generate_context_and_secret(poly_modulus_degree: int=4096, plain_modulus: int=1032193, global_scale_exp: int = 40) -> Tuple[ts.Context, ts_enc_context.SecretKey]:
    ckks_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree, plain_modulus=plain_modulus)
    ckks_context.global_scale = 2**global_scale_exp
    ckks_context.auto_rescale = False #It seems to decrease the used memory
    ckks_secret_key = ckks_context.secret_key()
    ckks_context.make_context_public()
    return ckks_context, ckks_secret_key

@profile
def create_ckks_encrypted_tensor_list(arrays: List[np.array], ckks_context: ts.Context) -> List[ts.CKKSTensor]:
    enc_tensors = [ts.ckks_tensor(ckks_context, array) for array in arrays]
    return enc_tensors

@profile
def decrypt_tensors(enc_tensors: List[ts.CKKSTensor], secret_key: ts_enc_context.SecretKey) -> List[np.array]:
    plain_tensors = [enc_tensor.decrypt(secret_key) for enc_tensor in enc_tensors]
    restructured_arrays = [np.array(plain_tensor.raw, dtype=np.float32).reshape(plain_tensor.shape) for plain_tensor in plain_tensors]
    return restructured_arrays