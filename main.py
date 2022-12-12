import optimizer
import train_nets
import speck as cipher

plain_bits = cipher.plain_bits
key_bits = cipher.key_bits
encryption_function = cipher.encrypt

def make_train_data(n, nr, delta_state, delta_key, _plain_bits, _key_bits, _encryption_function):
    return data # an n by _plain_bits*2 binary matrix

if __name__ == "__main__":
    # Find good input differences for SPECK
    best_differences, highest_round = optimizer.optimize(plain_bits, key_bits, encryption_function) # + other parameters
    delta_state, delta_key = best_differences[-1] # Getting the best input difference

    # Training the neural distinguisher, starting from 1 round before the last biased round detected by the optimizer
    best_round, best_val_acc = train_nets.train_neural_distinguisher(starting_round = max(1, highest_round-1),
                                                                     data_generator = lambda num_samples, num_rounds : make_train_data(num_samples, num_rounds, delta_state, delta_key))