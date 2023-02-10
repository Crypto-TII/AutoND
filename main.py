from os import urandom
from glob import glob
import ray
import importlib
import numpy as np
import os
import optimizer
import train_nets

num_runs = 1

def make_train_data(encryption_function, plain_bits, key_bits, n, nr, delta_state=0, delta_key=0):
    """TEMPORARY VERSION."""
    keys0 = (np.frombuffer(urandom(n*key_bits),dtype=np.uint8)&1)
    keys0 = keys0.reshape(n, key_bits);
    pt0 = (np.frombuffer(urandom(n*plain_bits),dtype=np.uint8)&1).reshape(n, plain_bits);
    keys1 = keys0^delta_key
    pt1 = pt0^delta_state
    C0 = encryption_function(pt0, keys0, nr)
    C1 = encryption_function(pt1, keys1, nr)
    C = np.hstack([C0, C1])
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    num_rand_samples = np.sum(Y==0);
    C[Y==0] = (np.frombuffer(urandom(num_rand_samples*plain_bits*2),dtype=np.uint8)&1).reshape(num_rand_samples, -1)
    # Sanity check
    assert C.shape[0] == n and C.shape[1] == 2*plain_bits
    return C, Y


@ray.remote(num_gpus=0.25)
def runBestDifferenceSearchForCipher(cipher_name, scenario, run):
    cipher = importlib.import_module('ciphers.' + cipher_name, package='ciphers')
    s = cipher.__name__[8:] + "_" + scenario
    print("\n")
    print("=" * 70)
    print("PART 1: Find the `best input difference` and the `highest round` using the evolutionary optimizer for ", s, "...")

    # Optimal difference search...
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    encryption_function = cipher.encrypt
    best_differences_bin, best_differences_int, highest_round = optimizer.optimize(plain_bits, key_bits, encryption_function, scenario = scenario, log_file=s, run = run) 

    print("\n")
    print("=" * 70)
    print(f"PART 2: Train DBitNet using staged training on the `best five input differences` starting two round before the `highest round`...")

    # Training the neural distinguisher, starting from two round before the last biased round detected by the optimizer
    for idx, delta in enumerate(best_differences_bin):
        if scenario == "related-key":
            delta_key = delta[plain_bits:]
        else:
            delta_key = 0
        delta_plain = delta[:plain_bits]
        print(s, hex(best_differences_int[idx]))
        # Starting at -2 because the optimizer now returns the first round for which nothing is found, rather than the last round for which something is found...
        best_round, best_val_acc = train_nets.train_neural_distinguisher(starting_round = max(1, highest_round-2),
                                                                 data_generator = lambda num_samples, nr : make_train_data(encryption_function, plain_bits, key_bits, num_samples, nr, delta_plain, delta_key))
        print(f'Best round for {s} : {best_round}, with accuracy {best_val_acc}')


if __name__ == "__main__":

    # Starting single and related key search for all ciphers in the ciphers folder
    modules = glob('ciphers/*.py')
    modules = [module[8:-3] for module in modules]
    print("List of ciphers to study : ", modules)
    L =[]
    for run in range(num_runs):
        for m in modules:
            if m.startswith('speck32'):
                L.append(runBestDifferenceSearchForCipher.remote(m, 'single-key', run))
                if not m.startswith('tea'): # Skipping tea related key, because it runs indefinitely due to the probability 1 related key trail.
                    L.append(runBestDifferenceSearchForCipher.remote(m, 'related-key', run))
    ray.get(L)

