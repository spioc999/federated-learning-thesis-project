from services.fed_learning import start_fed_averaging_mnist_simulation

#TODO add argparse
if __name__ == "__main__":
    start_fed_averaging_mnist_simulation(
        num_clients=10,
        num_rounds=10,
        fraction_fit=1.0,
        fraction_evaluate=0.3,
        enable_he=True,
        enable_zk_proof=False,
        verbose=True
    )