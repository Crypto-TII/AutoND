from os import urandom
import numpy as np

import optimizer
import train_nets
import speck3264 as cipher
#import speck64128 as cipher
#import speck128256 as cipher
#import simon3264 as cipher
#import simon64128 as cipher
#import simon128256 as cipher
#import present80 as cipher
#import tea as cipher
#import xtea as cipher

plain_bits = cipher.plain_bits
key_bits = cipher.key_bits
encryption_function = cipher.encrypt
scenario = "single-key"
#scenario = "related-key"

def make_train_data(n, nr, delta_state=0, delta_key=0):
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

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("PART 1: Find the `best input difference` and the `highest round` using the evolutionary optimizer...")
    ## Find good input differences for SPECK
    best_differences, highest_round = optimizer.optimize(plain_bits, key_bits, encryption_function, scenario = scenario)
    best_difference = best_differences[-1]
    if scenario == "related-key":
        delta_key = best_difference[plain_bits:]
    else:
        delta_key = 0
    delta_plain = best_difference[:plain_bits]

    print("\n")
    print("=" * 70)
    print(f"PART 2: Train DBitNet using staged training on the `best input difference` starting one round before the `highest round`...")

    # Training the neural distinguisher, starting from 1 round before the last biased round detected by the optimizer
    best_round, best_val_acc = train_nets.train_neural_distinguisher(starting_round = max(1, highest_round-1),
                                                                     data_generator = lambda num_samples, num_rounds : make_train_data(num_samples, num_rounds, delta_plain, delta_key))
