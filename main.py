import os

from os import urandom
from glob import glob
import ray
import importlib
import numpy as np
import os
import optimizer
import train_nets
import argparse


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


def runBestDifferenceSearchForCipher(cipher_name, scenario, output_dir):
    cipher = importlib.import_module('ciphers.' + cipher_name, package='ciphers')
    s = cipher.__name__[8:] + "_" + scenario
    print("\n")
    print("=" * 70)
    print("PART 1: Find the `best input difference` and the `highest round` using the evolutionary optimizer for ", s, "...")

    # Optimal difference search...
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    word_size = cipher.word_size
    encryption_function = cipher.encrypt
    best_differences_bin, best_differences_int, highest_round = optimizer.optimize(plain_bits, key_bits, encryption_function, scenario = scenario, log_file=f'{output_dir}/{s}') 

    print("\n")
    print("=" * 70)
    print(f"PART 2: Train DBitNet using staged training on the `best input differences` starting two round before the `highest round`...")

    # Training the neural distinguisher, starting from two round before the last biased round detected by the optimizer
    for idx, delta in enumerate(best_differences_bin):
        if scenario == "related-key":
            delta_key = delta[plain_bits:]
        else:
            delta_key = 0
        delta_plain = delta[:plain_bits]
        print(s, hex(best_differences_int[idx]))
        # Starting at -2 because the optimizer now returns the first round for which nothing is found, rather than the last round for which something is found...
        best_round_gohr, best_val_acc_gohr = train_nets.train_neural_distinguisher(starting_round = max(1, highest_round-2),
                                                                 data_generator = lambda num_samples, nr : make_train_data(encryption_function, plain_bits, key_bits, num_samples, nr, delta_plain, delta_key), model_name = 'gohr', input_size = plain_bits, word_size = word_size, log_prefix = f'{output_dir}/{s}_{hex(best_differences_int[idx])}')
        best_round_dbitnet, best_val_acc_dbitnet = train_nets.train_neural_distinguisher(starting_round = max(1, highest_round-2),
                                                                 data_generator = lambda num_samples, nr : make_train_data(encryption_function, plain_bits, key_bits, num_samples, nr, delta_plain, delta_key), model_name = 'dbitnet', input_size = plain_bits, word_size = word_size, log_prefix = f'{output_dir}/{s}_{hex(best_differences_int[idx])}')
        with open(f'{output_dir}/{s}_{hex(best_differences_int[-1])}_final', 'a') as f:
            f.write('Best difference, highest Gohr round, gohr val acc, highest dbitnet round, dbitnet val acc\n')
            f.write(f'[({hex(best_differences_int[-1]>>key_bits)}, {hex(best_differences_int[-1]&(2**key_bits-1))})], {best_round_gohr}, {best_val_acc_gohr}, {best_round_dbitnet}, {best_val_acc_dbitnet}')




if __name__ == "__main__":
    ciphers_list = glob('ciphers/*.py')
    ciphers_list = [cipher[8:-3] for cipher in ciphers_list]
    scenarios_list = ['single-key', 'related-key']
    parser = argparse.ArgumentParser(description='Obtain good input differences for neural cryptanalysis.')
    parser.add_argument('cipher', type=str, nargs=1,
            help=f'the name of the cipher to be analyzed, from the following list: {ciphers_list}')
    parser.add_argument('scenario', type=str, nargs=1,
            help=f'the scenario, either single-key or related-key', default = 'single-key')
    parser.add_argument('-o', '--output', type=str, nargs=1, default ='results',
            help=f'the folder where to store the experiments results')
    arguments = parser.parse_args()
    cipher_to_study = arguments.cipher[0]
    scenario = arguments.scenario[0]
    output_dir = arguments.output[0]
    if cipher_to_study not in ciphers_list:
        raise Exception(f'Cipher name error: it has to be one of {ciphers_list}.')
    if scenario not in scenarios_list:
        raise Exception(f'Scenario name error: it has to be one of {scenarios_list}.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Running the search for {cipher_to_study}, in the {scenario} scenario...')
    runBestDifferenceSearchForCipher(cipher_to_study, scenario, output_dir)
