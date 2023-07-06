# ------------------------------------------------
# TensorFlow import
# ------------------------------------------------
# TensorFlow setting: Which GPU to use and not to consume the whole GPU:
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'            # Which GPU to use.
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'            # Filters TensorFlow warnings.
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    # Prevents TensorFlow from consuming the whole GPU.
# Import TensorFlow:
import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ------------------------------------------------
# Other imports
# ------------------------------------------------
import logging
import pandas as pd
import numpy as np
from dbitnet import make_model as make_dbitnet
from gohrnet import make_model as make_gohrnet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam


# ------------------------------------------------
# Configuration and constants
# ------------------------------------------------
logging.basicConfig(level=logging.FATAL)

ABORT_TRAINING_BELOW_ACC = 0.5025   # if the validation accuracy reaches or falls below this limit, abort further training.
EPOCHS = 5                        # train for 10 epochs
NUM_SAMPLES = 10**7                 # create 10 million training samples
NUM_VAL_SAMPLES = 10**6             # create 1 million validation samples
BATCHSIZE = 5000                    # training batch size

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);



def train_one_round(model,
                    X, Y, X_val, Y_val,
                    round_number: int,
                    epochs=40,
                    model_name = 'model',
                    load_weight_file=False,
                    log_prefix = '',
                    LR_scheduler = None):
    """Train the `model` on the training data (X,Y) for one round.

    :param model: TensorFlow neural network
    :param X, Y: training data
    :param X_val, Y_val: validation data
    :param epochs: number of epochs to train
    :param load_weight_file: Boolean (if True: load weights from previous round.)
    :return: best validation accuracy
    """
    #------------------------------------------------
    # Handle model weight checkpoints
    #------------------------------------------------
    from tensorflow.keras.callbacks import ModelCheckpoint

    # load weight checkpoint from previous round?
    if load_weight_file:
        logging.info("loading weights from previous round...")
        model.load_weights(f'{log_prefix}_{model_name}_round{round_number-1}.h5')

    # create model checkpoint callback for this round
    checkpoint = ModelCheckpoint(f'{log_prefix}_{model_name}_round{round_number}.h5', monitor='val_loss', save_best_only = True)
    if LR_scheduler == None:
        callbacks = [checkpoint]
    else:
        callbacks = [checkpoint, LR_scheduler]



    #------------------------------------------------
    # Train the model
    #------------------------------------------------
    history = model.fit(X, Y, epochs=epochs, batch_size=BATCHSIZE,
                        validation_data=(X_val, Y_val), callbacks=callbacks, verbose = 0)



    # save the training history
    pd.to_pickle(history.history, f'{log_prefix}_{model_name}_training_history_round{round_number}.pkl')
    return np.max(history.history['val_acc'])

def train_neural_distinguisher(starting_round, data_generator, model_name, input_size, word_size, log_prefix = './', _epochs = EPOCHS, _num_samples=None):
    """Staged training of model_name starting in `starting_round` for a cipher with data generated by `data_generator`.

    :param starting_round:  Integer in which round to start the neural network training.
    :param data_generator:  Data_generator(number_of_samples, current_round) returns X, Y.
    :return: best_round, best_val_acc
    """

    #------------------------------------------------
    # Create the neural network model
    #------------------------------------------------
    logging.info(f'CREATE NEURAL NETWORK MODEL {model_name}')
    lr = None
    if model_name == 'dbitnet':
        model = make_dbitnet(2*input_size)
        optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    elif model_name == 'gohr':
        model = make_gohrnet(2*input_size, word_size=word_size)
        lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
        optimizer = 'adam'
    elif model_name == 'gohr_amsgrad':
        model = make_gohrnet(2*input_size, word_size=word_size)
        optimizer = Adam(amsgrad=True)

    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

    #------------------------------------------------
    # Start staged training from starting_round
    #------------------------------------------------
    current_round = starting_round
    load_weight_file = False
    best_val_acc = None
    best_round = None
    #------------------------------------------------
    # Using custom parameters if needed
    #------------------------------------------------
    if _epochs == None:
        epochs = EPOCHS
    else:
        epochs = _epochs
    if _num_samples == None:
        num_samples = NUM_SAMPLES
    else:
        num_samples = _num_samples

    print(f'Training on {epochs} epochs ...')
    while True:
        # ------------------------------------------------
        # Train one round
        # ------------------------------------------------
        # create data
        logging.info(f"CREATE CIPHER DATA for round {current_round} (training samples={num_samples:.0e}, validation samples={NUM_VAL_SAMPLES:.0e})...")
        X, Y = data_generator(NUM_SAMPLES, current_round)
        X_val, Y_val = data_generator(NUM_VAL_SAMPLES, current_round)

        # train model for the current round
        logging.info(f"TRAIN neural network for round {current_round}...")
        val_acc = train_one_round(model,
                                    X, Y, X_val, Y_val,
                                    current_round,
                                    epochs = epochs,
                                    load_weight_file = load_weight_file,
                                    log_prefix = log_prefix,
                                    model_name = model_name,
                                    LR_scheduler = lr)
        print(f'{model_name}, round {current_round}. Best validation accuracy: {val_acc}', flush=True)

        # after the starting_round, load the weight files:
        load_weight_file = True

        # abort further training if the validation accuracy is too low
        if val_acc <= ABORT_TRAINING_BELOW_ACC:
            logging.info(f"ABORT TRAINING (best validation accuracy {val_acc}<={ABORT_TRAINING_BELOW_ACC}).")
            break
        # otherwise save results as currently best reached round
        else:
            best_round = current_round
            best_val_acc = val_acc
            current_round += 1
        tf.keras.backend.clear_session()

    return best_round, best_val_acc
