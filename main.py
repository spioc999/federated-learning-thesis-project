from services.fed_learning import start_fed_averaging_mnist_simulation
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='2-way trusted FedAveraging',
                    description='Thesis project by Caronia Simone Pio')
    parser.add_argument('-he', '--homomorphic_encryption',
                    action='store_true')
    parser.add_argument('-zk', '--zk_proof',
                    action='store_true')
    parser.add_argument('-dv', '--disable_verbose',
                    action='store_true')
    return parser.parse_args()


if __name__ == "__main__":    
    args = parse_args()

    start_fed_averaging_mnist_simulation(
        num_clients=10,
        num_rounds=5,
        fraction_fit=1.0,
        fraction_evaluate=0.3,
        enable_he=args.homomorphic_encryption,
        enable_zk_proof=args.zk_proof,
        verbose=not args.disable_verbose
    )