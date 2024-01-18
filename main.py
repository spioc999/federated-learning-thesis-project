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
    parser.add_argument('-c', '--num_clients', required=False, default=5)
    parser.add_argument('-r', '--num_rounds', required=False, default=5)
    parser.add_argument('-f', '--fraction_fit', required=False, default=1.0)
    parser.add_argument('-e', '--fraction_evaluate', required=False, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":    
    args = parse_args()

    start_fed_averaging_mnist_simulation(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        enable_he=args.homomorphic_encryption,
        enable_zk_proof=args.zk_proof,
        verbose=not args.disable_verbose
    )