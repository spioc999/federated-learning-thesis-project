import tenseal as ts
import numpy as np

def generate_context_and_secret():
    ckks_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, plain_modulus=1032193)
    ckks_context.global_scale = 2**40
    ckks_secret_key = ckks_context.secret_key()
    ckks_context.make_context_public()
    return ckks_context, ckks_secret_key


def decrypt_aggregated_weights_and_scale(ndarrays: list, secret_key, scale: int) -> list:
    ndarrays_dec = [enc_tensor.decrypt(secret_key) for enc_tensor in ndarrays]
    ndarrays_restructured = [np.array(plain_tensor.raw, dtype=np.float32).reshape(plain_tensor.shape) for plain_tensor in ndarrays_dec]
    ndarrays_scaled = [ndarray / scale for ndarray in ndarrays_restructured]
    return ndarrays_scaled