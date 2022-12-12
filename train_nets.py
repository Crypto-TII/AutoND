# TensorFlow setting: Which GPU to use and not to consume the whole GPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'            # Which GPU to use.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'            # Filters TensorFlow warnings.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    # Prevents TensorFlow from consuming the whole GPU.

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import pandas as pd

from .dbitnet import make_model

# CONSTANTS:
ABORT_TRAINING_BELOW_ACC = 0.5050   # if the validation accuracy reaches or falls below this limit, abort further training.
EPOCHS = 40                         # train for 40 epochs
NUM_SAMPLES = 10e6                  # create 10 million training samples
NUM_VAL_SAMPLES = 1e6               # create 1 million validation samples
BATCHSIZE = 5000                    # training batch size

def train_one_round(model,
                    X, Y, X_val, Y_val,
                    round_number: int,
                    epochs=40,
                    load_weight_file=False):
    """Train the `model` on the training data (X,Y) for one round.

    :param model: TensorFlow neural network
    :param X, Y: training data
    :param X_val, Y_val: validation data
    :param epochs: number of epochs to train
    :param load_weight_file: Boolean (if True: load weights from previous round.)
    :return: best validation accuracy
    """
    #------------------------------------------------
    # handle model weight checkpoints
    #------------------------------------------------
    from keras.callbacks import ModelCheckpoint

    # load weight checkpoint from previous round?
    if load_weight_file:
        model.load_weights(f'results/model_round{round_number}.h5')

    # create model checkpoint callback for this round
    checkpoint = ModelCheckpoint(f'results/model_round{round_number}.h5', monitor='val_loss', save_best_only = True)

    #------------------------------------------------
    # train the model
    #------------------------------------------------
    history = model.fit(X, Y, epochs=epochs, batch_size=BATCHSIZE,
                        validation_data=(X_val, Y_val), callbacks=[checkpoint],
                        verbose=True)

    print("Best validation accuracy: ", np.max(history.history['val_acc']))

    # save the training history
    pd.to_pickle(history.history, f'results/training_history_round{round_number}.pkl')
    return np.max(history.history['val_acc'])

def train_neural_distinguisher(starting_round, data_generator):
    """

    :param starting_round:
    :param data_generator:
    :return: best_round, best_val_acc
    """

    #------------------------------------------------
    # create the neural network model
    #------------------------------------------------
    _X, _Y = data_generator(1, starting_round)  # create a single datapoint to determine the input size
    input_size = _X.shape[0]                    # determine the input_size from the single datapoint
    print("DETERMINED INPUT SIZE = ", input_size) # TODO: remove line this once debugged
    model = make_model(input_size)
    optimizer = tf.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

    #------------------------------------------------
    # start staged training from starting_round
    #------------------------------------------------
    current_round = starting_round
    load_weight_file = False
    best_val_acc = None
    best_round = None

    while True:

        # create data
        X, Y = data_generator(NUM_SAMPLES, current_round)
        X_val, Y_val = data_generator(NUM_VAL_SAMPLES, current_round)

        # train model for the current round
        train_one_round(model,
                        X, Y, X_val, Y_val,
                        current_round,
                        epochs = EPOCHS,
                        load_weight_file = load_weight_file)

        # after the starting_round, load the weight files:
        load_weight_file = True

        # abort further training if the validation accuracy is too low
        if val_acc <= ABORT_TRAINING_BELOW_ACC:
            break
        # otherwise save results as currently best reached round
        else:
            best_round = current_round
            best_val_acc = val_acc
            current_round += 1

    return best_round, best_val_acc