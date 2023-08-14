import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import tensorflow as tf
import numpy as np

import logging
import glob
import argparse
import importlib

import main as autond

# ------------------------------------------------
# Configuration and constants
# ------------------------------------------------

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

nEval = 5 # number of evaluation datasets which are freshly generated
num_val_samples = 10**6 # number of samples in each of the evaluation datasets
batchsize = 5000

def evaluate_X_Y(model, X, Y):
    """Returns the accuracy, TPR, and TNR of the model(X) with ground-truth Y.
    (Taken from Gohr's repository)
    """
    # ------------ calculate TPR and TNR
    Z = model.predict(X, batch_size=batchsize)
    Z = Z.flatten();
    Zbin = (Z > 0.5);
    # diff = Y - Z;
    # mse = np.mean(diff * diff);
    n = len(Z);
    n0 = np.sum(Y == 0);
    n1 = np.sum(Y == 1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y == 1]) / n1;
    tnr = np.sum(Zbin[Y == 0] == 0) / n0;
    return acc, tpr, tnr

def evaluate_Xlist_Ylist(model, Xlist, Ylist):

    allAccs, allTPRs, allTNRs = [], [], []

    for X, Y in zip(Xlist, Ylist):

        acc, tpr, tnr = evaluate_X_Y(model, X, Y)

        allAccs.append(acc)
        allTPRs.append(tpr)
        allTNRs.append(tnr)

        logging.info(f"\t acc={acc:.4f} \t tpr={tpr:.4f} \t tnr={tnr:.4f}")

    return allAccs, allTPRs, allTNRs

def get_deltas_from_scenario(scenario, input_difference, plain_bits, key_bits):
    """Returns delta_plain, delta_key for the scenario."""
    if scenario == "related-key":
        delta = autond.integer_to_binary_array(input_difference, plain_bits + key_bits)
        delta_key = delta[:, plain_bits:]
    elif scenario == "single-key":
        delta = autond.integer_to_binary_array(input_difference, plain_bits)
        delta_key = 0
    else:
        raise ValueError(f"An unknown scenario '{scenario}' was encountered.")
    delta_plain = delta[:, :plain_bits]
    return delta_plain, delta_key

def parseTheCommandLine(parser):
    # ---- add arguments to parser
    # model arguments
    parser.add_argument('--model_path',
                        type=str,
                        required=False,
                        help=f'The path to the h5 file with the model weights.')
    parser.add_argument('--model_type',
                        type=str,
                        default='dbitnet',
                        required=False,
                        choices=['gohr-depth1', 'dbitnet'],
                        help=f'The model type (gohr-depth1 or dbitnet) of the model h5 file.')

    # cipher arguments
    parser.add_argument('--cipher',
                        type=str,
                        default='speck3264',
                        required=True,
                        help=f'The name of the cipher to be analyzed, from the following list: {ciphers_list}.')
    parser.add_argument('--scenario',
                        type=str,
                        required=False,
                        choices=['single-key', 'related-key'],
                        help=f'The scenario, either single-key or related-key',
                        default = 'single-key')

    # dataset arguments
    parser.add_argument('--dataset_path_X',
                        #type=str,
                        required=False,
                        nargs='+',
                        help=f'Optional path to a pre-existing *.npy dataset file TODO format-hint')
    parser.add_argument('--dataset_path_Y',
                        #type=str,
                        required=False,
                        nargs='+',
                        help=f'Optional path to a pre-existing *.npy dataset file TODO format-hint')
    parser.add_argument('--input_difference',
                        type=str,
                        required=False,
                        help=f"The input difference for the data generation, e.g. '0x40'.")
    parser.add_argument('--round_number',
                        type=str,
                        required=False,
                        help=f"The round number for the data generation, e.g. 5.")

    # results arguments
    parser.add_argument('--add_str',
                        type=str,
                        required=False,
                        default='',
                        help=f'Add an additional string to the evaluation filename.')
    # ---------------------------------------------------
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # create a list of available ciphers in the ciphers folder:
    ciphers_list = glob.glob('ciphers/*.py')
    ciphers_list = [cipher[8:-3] for cipher in ciphers_list]

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Evaluate an existing neural distinguisher.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parseTheCommandLine(parser)

    # ---------------------------------------------------
    # infer from the command line arguments:
    cipher = importlib.import_module('ciphers.' + args.cipher, package='ciphers')
    input_size = cipher.plain_bits

    # create the filename for the results (the same path as the model, just with ending '_eval.npz')
    filename_results = args.model_path.replace('.h5', f'_eval{args.add_str}.npz')

    # ---------------------------------------------------
    logging.info(f"Creating model '{args.model_type}' with weights from path \n\t '{args.model_path}'...")
    if args.model_type == 'gohr-depth1':
        import gohrnet
        model = gohrnet.make_model(2*input_size, word_size=cipher.word_size)
    elif args.model_type == 'dbitnet':
        import dbitnet
        model = dbitnet.make_model(2*input_size)
    else:
        logging.fatal(f"Model creation failed for model_type={args.model_type}.")
        exit()

    # the optimizer and loss don't really matter for the evaluation we do here:
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    model.load_weights(args.model_path)

    Xlist, Ylist = [], []

    # ---------------------------------------------------
    if (args.dataset_path_X is not None) & (args.dataset_path_Y is not None):

        for X, Y in zip(args.dataset_path_X, args.dataset_path_Y):
            logging.info(f"Loading datasets \n\t X={X} \n\t Y={Y}")
            Xlist.append(np.load(X))
            Ylist.append(np.load(Y))

    elif (args.input_difference is not None) & (args.round_number is not None):
        logging.info(f"""Creating new validation dataset for 
                        cipher: {args.cipher}, 
                        scenario: {args.scenario}, 
                        input difference: {args.input_difference}, 
                        round number: {args.round_number}, 
                        cipher.plain_bits: {cipher.plain_bits}, 
                        cipher.key_bits: {cipher.key_bits}
                        ...""")

        # input_difference: convert str to hexadecimal int
        args.input_difference = int(args.input_difference, base=16)
        args.round_number = int(args.round_number)

        delta_plain, delta_key = get_deltas_from_scenario(args.scenario,
                                                          args.input_difference,
                                                          cipher.plain_bits,
                                                          cipher.key_bits)

        data_generator = lambda num_samples, nr: autond.make_train_data(cipher.encrypt,
                                                                         cipher.plain_bits,
                                                                         cipher.key_bits,
                                                                         num_samples,
                                                                         nr,
                                                                         delta_plain,
                                                                         delta_key)

        for i in range(nEval):
            X, Y = data_generator(num_val_samples, args.round_number)
            Xlist.append(X)
            Ylist.append(Y)

    else:
        logging.fatal("Please, pass one of the arguments: 'data_path', ('input_difference' and 'round_number').")
        exit()

    # ---------------------------------------------------
    logging.info(f"Running evaluations...")
    accs, tprs, tnrs = evaluate_Xlist_Ylist(model, Xlist, Ylist)

    # ---------------------------------------------------
    logging.info(f"Saving results to {filename_results}...")
    np.savez(filename_results,
             accs=accs,
             tprs=tprs,
             tnrs=tnrs)
