def get_dilation_rates(input_size):
    """Helper function to determine the dilation rates of DBitNet given an input_size. """
    drs = []
    while input_size >= 8:
        drs.append(int(input_size / 2 - 1))
        input_size = input_size // 2

    return drs

def make_model(input_size=64, n_filters=32, n_add_filters=16):
    """Create a DBITNet model.

    :param input_size: e.g. for SPECK32/64 the input_size is 64 bit.
    :return: DBitNet model.
    """

    import tensorflow as tf
    from tf.keras.models import Model
    from tf.keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate, BatchNormalization, Activation, Add
    from tf.keras.regularizers import l2

    # determine the dilation rates from the given input size
    dilation_rates = get_dilation_rates(input_size)

    # prediction head parameters (similar to Gohr)
    d1 = 256 # TODO this can likely be reduced to 64.
    d2 = 64
    reg_param = 1e-5

    # define the input shape
    inputs = Input(shape=(input_size, 1))
    x = inputs

    # normalize the input data to a range of [-1, 1]:
    x = tf.subtract(x, 0.5)
    x = tf.divide(x, 0.5)

    for dilation_rate in dilation_rates:
        ### wide-narrow blocks
        x = Conv1D(filters=n_filters,
                   kernel_size=2,
                   padding='valid',
                   dilation_rate=dilation_rate,
                   strides=1,
                   activation='relu')(x)
        x = BatchNormalization()(x)
        x_skip = x
        x = Conv1D(filters=n_filters,
                   kernel_size=2,
                   padding='causal',
                   dilation_rate=1,
                   activation='relu')(x)
        x = Add()([x, x_skip])
        x = BatchNormalization()(x)

        n_filters += n_add_filters

    ### prediction head
    out = tf.keras.layers.Flatten()(x)

    dense0 = Dense(d1, kernel_regularizer=l2(reg_param))(out);
    dense0 = BatchNormalization()(dense0);
    dense0 = Activation('relu')(dense0);
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)

    model = Model(inputs, out)

    return model

