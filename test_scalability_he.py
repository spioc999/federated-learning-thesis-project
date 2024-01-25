from services.ckks_he import * 
from services.keras_and_datasets import get_model
from services.logger import setup_logger, log_info
import argparse
from memory_profiler import profile

@profile
def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num')
    args = parser.parse_args()
    setup_logger()
    NUM_SUM = int(args.num)

    model_weights = get_model().get_weights()

    log_info(f'Start create context')
    context, secret = generate_context_and_secret()
    log_info(f'End create context')

    log_info(f'Start encrypting weights, num {NUM_SUM}')

    clients_weights = [create_ckks_encrypted_tensor_list(model_weights, context) for _ in range(NUM_SUM)]

    log_info(f'End encrypting weights')

    log_info(f'Start sum')
    summed_weights = [client_weights for client_weights in clients_weights[0]]
    for client_weights in clients_weights[1:]:
        for i, weights in enumerate(client_weights):
            summed_weights[i] = summed_weights[i] + weights
    log_info(f'End sum')

    del clients_weights

    log_info(f'Start decrypt')
    decrypted_weights = decrypt_tensors(summed_weights, secret)
    log_info(f'End decrypt')

    log_info(f'Start scale')
    scaled_weights = [arrays / NUM_SUM for arrays in decrypted_weights]
    log_info(f'End scale')

    del summed_weights
    del decrypted_weights
    del scaled_weights

if __name__ == "__main__":
    run_main()

