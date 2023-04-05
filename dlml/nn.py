
import os
import json
import signal
from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, callbacks
import tensorflow_addons as tfa

from .utils import print_msg, print_warning


__all__ = ['LEARNING_RATE', 'LearningRateCallback', 'make_preprocessing_pipeline_1D',
           'make_preprocessing_pipeline_2D', 'build_model', 'train_model', 'predict',
           'sigint_handler', 'SpectralPooling', 'DownSampling1D', 'MaxPooling1DWithArgmax',
           'compute_receptive_field', 'compute_correlations']

LEARNING_RATE = []


class SpectralPooling(keras.layers.Layer):
    def __init__(self, sampling_rate, cutoff_frequency, **kwargs):
        super(SpectralPooling, self).__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.cutoff_frequency = cutoff_frequency

    @property
    def dt(self):
        return self.__dt

    @property
    def sampling_rate(self):
        return self.__sampling_rate
    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        if sampling_rate < 0:
            raise ValueError('Sampling rate must be >= 0')
        self.__sampling_rate = sampling_rate
        self.__dt = 1 / sampling_rate

    @property
    def cutoff_frequency(self):
        return self.__cutoff_frequency
    @cutoff_frequency.setter
    def cutoff_frequency(self, cutoff_frequency):
        if cutoff_frequency < 0:
            raise ValueError('Cut-off frequency must be >= 0')
        self.__cutoff_frequency = cutoff_frequency

    def build(self, input_shape):
        n_steps = input_shape[1] // 2 + 1
        freq = tf.linspace(0., self.sampling_rate / 2., n_steps)
        idx = tf.squeeze(tf.where(freq <= self.cutoff_frequency))
        self.idx = slice(0, idx[-1])
        self.coeff = self.cutoff_frequency / self.sampling_rate * 2

    def call(self, inputs):
        perm = 0, 2, 1
        # tf.signal.rfft works on the inner most dimension, i.e., the last one
        x = tf.transpose(inputs, perm)
        xf = tf.signal.rfft(x)
        xf_trunc = xf[:, :, self.idx]
        x_sub = tf.transpose(tf.signal.irfft(xf_trunc) * self.coeff, perm)
        return x_sub

    def get_config(self):
        return {'sampling_rate': self.sampling_rate, 'cutoff_frequency': self.cutoff_frequency}


class DownSampling1D(keras.layers.Layer):
    def __init__(self, steps=2, **kwargs):
        super(DownSampling1D, self).__init__(**kwargs)
        self.steps = steps

    @property
    def steps(self):
        return self.__steps
    @steps.setter
    def steps(self, steps):
        if steps <= 0:
            raise ValueError('Downsampling steps must be > 0')
        self.__steps = steps

    def build(self, input_shape):
        self.idx = slice(0, input_shape[1], self.steps)

    def call(self, inputs):
        return inputs[:, self.idx, :]

    def get_config(self):
        return {'steps': self.steps}


class MaxPooling1DWithArgmax(keras.layers.MaxPooling1D):
    def __init__(self, pool_size=2, strides=None,
                 padding='valid', data_format='channels_last',
                 store_argmax=False, **kwargs):

        super(MaxPooling1DWithArgmax, self).__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )
        self.store_argmax = store_argmax
        self.padding_upper = padding.upper()

    def call(self, inputs):
        if self.store_argmax:
            ret = tf.nn.max_pool_with_argmax(tf.expand_dims(inputs, 1),
                                             ksize=(1, self.pool_size[0]),
                                             strides=(1, self.strides[0]),
                                             padding=self.padding_upper)
            self.argmax = ret.argmax
        return super(MaxPooling1DWithArgmax, self).call(inputs)

    def get_config(self):
        return {'pool_size': self.pool_size[0], 'strides': self.strides[0],
                'padding': self.padding, 'data_format': self.data_format,
                'store_argmax': self.store_argmax}


TERMINATE_TF = False
def sigint_handler(sig, frame):
    global TERMINATE_TF
    if sig == signal.SIGINT:
        first = True
        while True:
            if first:
                ans = input('\nTerminate training at the end of the current training epoch? [yes/no] ').lower()
                first = False
            if ans == 'yes':
                TERMINATE_TF = True
                break
            elif ans == 'no':
                break
            else:
                ans = input('Please enter "yes" or "no": ').lower()


class SigIntHandlerCallback(keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, batch, logs=None):
        self.model.stop_training = TERMINATE_TF


class LearningRateCallback(keras.callbacks.Callback):
    def __init__(self, model, experiment = None):
        self.model = model
        self.experiment = experiment
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        try:
            lr = self.model.optimizer.learning_rate(self.step).numpy()
        except:
            lr = self.model.optimizer.learning_rate
        LEARNING_RATE.append(lr)
        if self.experiment is not None:
            self.experiment.log_metric('learning_rate', lr, self.step)


def make_preprocessing_pipeline_1D(input_layer, N_units, kernel_size, kernel_stride, activation_fun, activation_loc, sampling_rate, pooling_type, count=None):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function "{activation_fun}"')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    base_name = input_layer.name
    for n,(N_conv,N_pooling,sz,strd) in enumerate(zip(N_units['conv'], N_units['pooling'], kernel_size, kernel_stride)):
        conv_lyr_name = base_name + (f'_conv_{count}_{n+1}' if count is not None else f'_conv_{n+1}')
        activ_lyr_name = base_name + (f'_relu-{count}_{n+1}' if count is not None else f'_relu_{n+1}')
        pool_lyr_name = base_name + (f'_pool_{count}_{n+1}' if count is not None else f'_pool_{n+1}')
        if pooling_type is None:
            pooling_layer = None
        elif pooling_type.lower() == 'max':
            pooling_layer = layers.MaxPooling1D(N_pooling,  name=pool_lyr_name)
        elif pooling_type.lower() == 'argmax':
            pooling_layer = MaxPooling1DWithArgmax(N_pooling,  name=pool_lyr_name)
        elif pooling_type.lower() in ('avg','average'):
            pooling_layer = layers.AveragePooling1D(N_pooling,  name=pool_lyr_name)
        elif pooling_type.lower() in ('down', 'downsampling', 'downsample'):
            pooling_layer = DownSampling1D(N_pooling, name=pool_lyr_name)
        elif pooling_type.lower() == 'spectral':
            layer_sampling_rate = sampling_rate / (2 ** n)
            cutoff_frequency = layer_sampling_rate / (2 ** N_pooling)
            pooling_layer = SpectralPooling(layer_sampling_rate, cutoff_frequency, name=pool_lyr_name)
        else:
            raise Exception(f'Unknown pooling type "{pooling_type}"')
        try:
            L = layers.Conv1D(filters=N_conv, kernel_size=sz, strides=strd, activation=None, name=conv_lyr_name)(L)
        except:
            L = layers.Conv1D(filters=N_conv, kernel_size=sz, strides=strd, activation=None, name=conv_lyr_name)(input_layer)
        if activation_fun is not None:
            if activation_loc == 'after_conv':
                L = layers.ReLU(name=activ_lyr_name)(L)
                if pooling_layer is not None:
                    L = pooling_layer(L)
            else:
                if pooling_layer is not None:
                    L = pooling_layer(L)
                L = layers.ReLU(name=activ_lyr_name)(L)
        elif pooling_layer is not None:
            L = pooling_layer(L)
    return L


def make_preprocessing_pipeline_2D(input_layer, N_units, kernel_size, kernel_stride, activation_fun, activation_loc, count=None):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function {activation_fun}')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    base_name = input_layer.name
    for n,(N_conv,N_pooling,sz,strd) in enumerate(zip(N_units['conv'], N_units['pooling'], kernel_size, kernel_stride)):
        conv_lyr_name = base_name + (f'_conv_{count}_{n+1}' if count is not None else f'_conv_{n+1}')
        activ_lyr_name = base_name + (f'_relu_{count}_{n+1}' if count is not None else f'_relu_{n+1}')
        pool_lyr_name = base_name + (f'_pool_{count}_{n+1}' if count is not None else f'_pool_{n+1}')
        try:
            L = layers.Conv2D(N_conv, [sz, 2], strides=strd, padding='same', activation=None, name=conv_lyr_name)(L)
        except:
            L = layers.Conv2D(N_conv, [sz, 2], strides=strd, padding='same', activation=None, name=conv_lyr_name)(input_layer)
        if activation_fun is not None:
            if activation_loc == 'after_conv':
                L = layers.ReLU(name=activ_lyr_name)(L)
                L = layers.MaxPooling2D([N_pooling, 1], name=pool_lyr_name)(L)
            else:
                L = layers.MaxPooling2D([N_pooling, 1], name=pool_lyr_name)(L)
                L = layers.ReLU(name=activ_lyr_name)(L)
        else:
            L = layers.MaxPooling2D([N_pooling, 1], name=pool_lyr_name)(L)
    return L


def build_model(N_samples, steps_per_epoch, var_names, model_arch, N_outputs, streams_mode, \
                normalization_strategy, loss_fun_pars, optimizer_pars, lr_schedule_pars, sampling_rate):
    """
    Builds and compiles the model
    
    The basic network topology used here is taken from the following paper:
   
    George, D., & Huerta, E. A. (2018).
    Deep neural networks to enable real-time multimessenger astrophysics.
    Physical Review D, 97(4), 044039. http://doi.org/10.1103/PhysRevD.97.044039
    """

    N_dims = model_arch['N_dims']
    if N_dims not in (1,2):
        raise Exception('The number of dimensions of the data must be either 1 or 2')

    metrics = None
    loss_fun_name = loss_fun_pars['name'].lower()
    if loss_fun_name == 'mae':
        loss = losses.MeanAbsoluteError()
    elif loss_fun_name == 'mape':
        loss = losses.MeanAbsolutePercentageError()
    elif loss_fun_name == 'binarycrossentropy':
        try:
            loss = losses.BinaryCrossentropy(from_logits=loss_fun_pars['from_logits'])
        except:
            loss = losses.BinaryCrossentropy(from_logits=True)
        metrics = ['binary_crossentropy', 'acc']
    else:
        raise Exception('Unknown loss function: {}.'.format(loss_function))

    if lr_schedule_pars is not None and 'name' in lr_schedule_pars:
        if lr_schedule_pars['name'] == 'cyclical':
            # According to [1], "experiments show that it often is good to set stepsize equal to
            # 2 − 10 times the number of iterations in an epoch".
            #
            # [1] Smith, L.N., 2017, March.
            #     Cyclical learning rates for training neural networks.
            #     In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 464-472). IEEE.
            #
            step_sz = steps_per_epoch * lr_schedule_pars['factor']
            learning_rate = tfa.optimizers.Triangular2CyclicalLearningRate(
                initial_learning_rate = lr_schedule_pars['initial_learning_rate'],
                maximal_learning_rate = lr_schedule_pars['max_learning_rate'],
                step_size = step_sz)
            print_msg(f'Will use cyclical learning rate scheduling with a step size of {step_sz}.')
        elif lr_schedule_pars['name'] == 'exponential_decay':
            initial_learning_rate = lr_schedule_pars['initial_learning_rate']
            decay_steps = lr_schedule_pars['decay_steps']
            decay_rate = lr_schedule_pars['decay_rate']
            learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)
            print_msg('Will use exponential decay of learning rate.')
        else:
            raise Exception(f'Unknown learning rate schedule: {lr_schedule_pars["name"]}')
    else:
        learning_rate = optimizer_pars['learning_rate']

    optimizer_name = optimizer_pars['name'].lower()
    if optimizer_name == 'sgd':
        momentum = optimizer_pars['momentum'] if 'momentum' in optimizer_pars else 0.
        nesterov = optimizer_pars['nesterov'] if 'nesterov' in optimizer_pars else False
        optimizer = optimizers.SGD(learning_rate, momentum, nesterov)
    elif optimizer_name in ('adam', 'adamax', 'nadam'):
        beta_1 = optimizer_pars['beta_1'] if 'beta_1' in optimizer_pars else 0.9
        beta_2 = optimizer_pars['beta_2'] if 'beta_2' in optimizer_pars else 0.999
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate, beta_1, beta_2)
        elif optimizer_name == 'adamax':
            optimizer = optimizers.Adamax(learning_rate, beta_1, beta_2)
        else:
            optimizer = optimizers.Nadam(learning_rate, beta_1, beta_2)
    elif optimizer_name == 'adagrad':
        initial_accumulator_value = optimizer_pars['initial_accumulator_value'] if \
                                    'initial_accumulator_value' in optimizer_pars else 0.1
        optimizer = optimizers.Adagrad(learning_rate, initial_accumulator_value)
    elif optimizer_name == 'adadelta':
        rho = optimizer_pars['rho'] if 'rho' in optimizer_pars else 0.95
        optimizer = optimizers.Adadelta(learning_rate, rho)
    else:
        raise Exception('Unknown optimizer: {}.'.format(optimizer_name))

    N_units = model_arch['N_units']
    try:
        kernel_size = model_arch['kernel_size']
        kernel_stride = model_arch['kernel_stride']
        activation_fun = model_arch['preproc_activation']
        activation_loc = model_arch['activation_loc']
        CNN = True
    except:
        # will build a simple fully-connected neural network
        CNN = False

    ### figure out how data should be normalized
    if normalization_strategy not in ('batch', 'layer', 'training_set'):
        raise ValueError('normalization_strategy must be one of "batch", "layer" or "trainin_set"')

    batch_norm = normalization_strategy.lower() == 'batch'
    normalization_layer = normalization_strategy.lower() == 'layer'

    def make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, kernel_stride,
                                  activation_fun, activation_loc, sampling_rate, pooling_type, count):
        if N_dims == 1:
            L = []
            for input_layer in input_layers:
                lyr = make_preprocessing_pipeline_1D(input_layer, N_units, kernel_size,
                                                     kernel_stride, activation_fun, activation_loc,
                                                     sampling_rate, pooling_type, count)
                L.append(lyr)
        else:
            L = make_preprocessing_pipeline_2D(input_layers[0], N_units, kernel_size,
                                               kernel_stride, activation_fun, activation_loc, count)
        return L

    def make_dense_stream(L, N_units, N_outputs, model_arch, count):
        for i,n in enumerate(N_units['dense']):
            L = layers.Dense(n, activation='relu', name=f'fc_{count}_{i+1}')(L)
        if model_arch['dropout_coeff'] > 0:
            L = layers.Dropout(model_arch['dropout_coeff'], name=f'dropout_{count}')(L)
        output = layers.Dense(N_outputs, name=f'predictions_{count}')(L)
        return output

    def make_full_stream(input_layers, N_dims, N_units, kernel_size, kernel_stride, activation_fun,
                         activation_loc, N_outputs, model_arch, sampling_rate, pooling_type, count):
        L = make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, kernel_stride,
                                      activation_fun, activation_loc, sampling_rate, pooling_type, count)
        if isinstance(L, list):
            if len(L) == 1:
                L = L[0]
            else:
                L = layers.concatenate(L, name=f'concat_{count}')
        L = layers.Flatten(name=f'flatten_{count}')(L)
        return make_dense_stream(L, N_units, N_outputs, model_arch, count)

    if CNN:
        if type(streams_mode) not in (int, float):
            raise Exception('streams_mode must be an integer in the range [0,3]')
        if streams_mode not in (0, 1, 2, 3):
            raise ValueError('streams_mode must be in the range [0,3]')

    pooling_type = model_arch['pooling_type']

    if N_dims == 1:
        inputs = []
        input_layers = []
        for var_name in var_names:
            input_layer = keras.Input(shape=(N_samples, 1), name=var_name)
            inputs.append(input_layer)
            if batch_norm:
                input_layer = layers.BatchNormalization()(input_layer)
            elif normalization_layer:
                input_layer = layers.experimental.preprocessing.Normalization()(input_layer)
            input_layers.append(input_layer)
    else:
        inputs = keras.Input(shape=(N_samples, 2, 1), name='_'.join(var_names))
        if batch_norm:
            input_layer = layers.BatchNormalization()(inputs)
        elif normalization_layer:
            input_layer = layers.experimental.preprocessing.Normalization()(inputs)
        else:
            input_layer = inputs
        input_layers = [input_layer]

    if CNN:
        if streams_mode == 0:
            # just one stream
            outputs = [make_full_stream(input_layers, N_dims, N_units, kernel_size, kernel_stride, activation_fun,
                                        activation_loc, N_outputs, model_arch, sampling_rate, pooling_type, 1)]
        elif streams_mode == 1:
            # common preprocessing stream and then one stream of dense layers for each output
            L = make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, kernel_stride,
                                          activation_fun, activation_loc, sampling_rate, pooling_type, 1)
            if isinstance(L, list):
                if len(L) == 1:
                    L = L[0]
                else:
                    L = layers.concatenate(L, name='concat_1')
            L = layers.Flatten(name='flatten_1')(L)
            outputs = [make_dense_stream(L, N_units, 1, model_arch, i+1) for i in range(N_outputs)]

        elif streams_mode == 2:
            # one preprocessing stream for each output and then one common stream of dense layers
            L = [make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, kernel_stride, activation_fun,
                                           activation_loc, sampling_rate, pooling_type, i+1) for i in range(N_outputs)]
            if isinstance(L[0], list):
                if len(L[0]) == 1:
                    L = L[0][0]
                else:
                    L = layers.concatenate([l for LL in L for l in LL], name='concat_1')
            else:
                if len(L) == 1:
                    L = L[0]
                else:
                    L = layers.concatenate(L, name='concat_1')
            L = layers.Flatten(name='flatten_1')(L)
            outputs = make_dense_stream(L, N_units, N_outputs, model_arch, 1)

        elif streams_mode == 3:
            # one full stream for each output
            outputs = [make_full_stream(input_layers, N_dims, N_units, kernel_size, kernel_stride, activation_fun,
                                        activation_loc, 1, model_arch, sampling_rate, pooling_type, i+1) for i in range(N_outputs)]

    else:
        L = layers.concatenate(input_layers, name='concat_inputs')
        L = layers.Flatten(name='flatten_inputs')(L)
        outputs = [make_dense_stream(L, N_units, N_outputs, model_arch, 1)]

    model = keras.Model(inputs=inputs, outputs=outputs)
    if metrics is None:
        model.compile(optimizer=optimizer, loss=loss)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, optimizer, loss


def train_model(model, x, y,
                N_epochs,
                batch_size,
                steps_per_epoch,
                output_dir,
                experiment = None,
                callbacks_pars = None,
                verbose = 1):

    checkpoint_dir = output_dir + '/checkpoints'
    os.makedirs(checkpoint_dir)

    # create a callback that saves the model's weights
    checkpoint_cb = callbacks.ModelCheckpoint(filepath = os.path.join(checkpoint_dir, 'weights.h5'),
                                              save_weights_only = False,
                                              save_best_only = True,
                                              monitor = 'val_loss',
                                              verbose = verbose)
    print_msg('Added callback for saving weights at checkpoint.')

    cbs = [checkpoint_cb, LearningRateCallback(model, experiment), SigIntHandlerCallback(model)]
    print_msg('Added callback for logging learning rate.')
    print_msg('Added callback for terminating training upon SIGINT.')

    try:
        for cb_pars in callbacks_pars:
            if cb_pars['name'] == 'early_stopping':
                # create a callback that will stop the optimization if there is no improvement
                early_stop_cb = callbacks.EarlyStopping(monitor = cb_pars['monitor'],
                                                        patience = cb_pars['patience'],
                                                        verbose = verbose,
                                                        mode = cb_pars['mode'])
                cbs.append(early_stop_cb)
                print_msg('Added callback for early stopping.')
            elif cb_pars['name'] == 'reduce_on_plateau':
                lr_scheduler_cb = callbacks.ReduceLROnPlateau(monitor = cb_pars['monitor'],
                                                              factor = cb_pars['factor'],
                                                              patience = cb_pars['patience'],
                                                              verbose = verbose,
                                                              mode = cb_pars['mode'],
                                                              cooldown = cb_pars['cooldown'],
                                                              min_lr = cb_pars['min_learning_rate'])
                cbs.append(lr_scheduler_cb)
                print_msg('Added callback for reducing learning rate on plateaus.')
            else:
                raise Exception(f'Unknown callback: {cb_pars["name"]}')
    except:
        print_warning('Not adding callbacks.')

    if len(model.inputs) == 1:
        x_training = tf.squeeze(x['training'])
        x_validation = tf.squeeze(x['validation'])
    else:
        input_names = [inp.name.split(':')[0] for inp in model.inputs]
        x_training = {name: x['training'][i] for i,name in enumerate(input_names)}
        x_validation = {name: x['validation'][i] for i,name in enumerate(input_names)}

    return model.fit(x_training,
                     y['training'],
                     epochs = N_epochs,
                     batch_size = batch_size,
                     steps_per_epoch = steps_per_epoch,
                     validation_data = (x_validation, y['validation']),
                     verbose = verbose,
                     callbacks = cbs)


def predict(model, data_sliding, window_step, rolling_length=50):
    # window_step is in seconds
    import pandas as pd
    var_names = data_sliding.keys()
    if len(model.inputs) > 1:
        x = {var_name: tf.constant(data_sliding[var_name], dtype=tf.float32) for var_name in var_names}
    elif len(model.inputs) == 1 and len(data_sliding.keys()) > 1:
        x = tf.constant(list(data_sliding.values()), dtype=tf.float32)
        x = tf.transpose(x, perm=(1,2,0))
    else:
        x = tf.constant(data_sliding[var_names[0]], dtype=tf.float32)
    y = model.predict(x)
    n_samples, n_outputs = y.shape
    data = {f'inertia_{i}': y[:,i] for i in range(n_outputs)}
    H = pd.DataFrame(data).rolling(rolling_length).mean().to_numpy()
    time = np.arange(n_samples) * window_step
    return time, H, y


def compute_receptive_field(model, stop_layer='', include_stop_layer=False):
    '''
    Computes the effective receptive field size and the effective stride of a convolutional neural network.

    Parameters:
    model - the CNN: it must be an instance of either keras.Model or keras.Sequential.
    stop_layer - the layer at which the computation of effective receptive field and stride should be stopped.
    It can be the layer itself, a string containing the name of the layer or a class. In the latter case,
    the computation will stop at the first instance of the given class. Pass an empty string to compute RF size
    and stride of the whole model.

    Returns:
    effective_RF_size - the effective receptive field size: a dictionary containing as keys the layer names
    and as values the corresponding RF size on the input
    effective_stride - the effective stride: a dictionary containing as keys the layer names and as values
    the corresponding stride on the input
    '''

    if not isinstance(model, keras.Model) and not isinstance(model, keras.Sequential):
        raise ValueError('Argument `model` must be an instance of either keras.Model or keras.Sequential')

    import inspect
    if isinstance(stop_layer, str):
        stop_layer_name = stop_layer
    elif isinstance(stop_layer, keras.layers.Layer):
        stop_layer_name = stop_layer.name
    elif inspect.isclass(stop_layer):
        stop_layer_name = None
    else:
        raise ValueError('Argument `stop_layer` should be a layer name, a layer instance or a layer class')

    def is_stop_layer(layer):
        return (stop_layer_name is not None and layer.name == stop_layer_name) or \
            (stop_layer_name is None and isinstance(layer, stop_layer))

    def find_next(layers, layer_name):
        for i,lyr in enumerate(layers):
            if len(lyr['inbound_nodes']) == 0:
                # an input layer: continue
                continue
            inbound_nodes_names = [node[0] for node in lyr['inbound_nodes'][0]]
            if layer_name in inbound_nodes_names:
                return lyr,layers[i:]
        return None,None

    def extract_layer_sequence(layers, layer_idx):
        next_layer,remaining_layers = layers[layer_idx], layers
        seq = [next_layer['name']]
        while True:
            next_layer,remaining_layers = find_next(remaining_layers, next_layer['name'])
            if next_layer is None:
                break
            seq.append(next_layer['name'])
        return seq

    def find_layer(model, layer_name):
        for layer in model.layers:
            if layer.name == layer_name:
                return layer
        return None

    arch = json.loads(model.to_json())
    layers = arch['config']['layers']
    input_layers_idx = [i for i in range(len(layers)) if len(layers[i]['inbound_nodes']) == 0]
    sequences = [extract_layer_sequence(layers, idx) for idx in input_layers_idx]

    ### compute the effective receptive field size
    effective_RF_size = {}
    for sequence in sequences:
        R_prev = 1
        for k,layer_name in enumerate(sequence):
            layer = find_layer(model, layer_name)
            if not include_stop_layer and is_stop_layer(layer):
                break
            if hasattr(layer, 'kernel_size'):
                # a convolutional layer
                if len(layer.kernel_size) > 1:
                    raise Exception('This function only works with 1D convolutional layers')
                fk = layer.kernel_size[0]
            elif hasattr(layer, 'pool_size'):
                # a pooling layer
                if len(layer.pool_size) > 1:
                    raise Exception('This function only works with 1D pooling layers')
                fk = layer.pool_size[0]
            else:
                fk = None
            if fk is not None:
                strides = []
                for i in range(1, k):
                    try:
                        strides.append(model.layers[i].strides[0])
                    except:
                        pass
                effective_RF_size[layer.name] = R_prev + (fk - 1) * np.prod(strides, dtype=np.int32)
            else:
                effective_RF_size[layer.name] = R_prev
            R_prev = effective_RF_size[layer.name]
            if is_stop_layer(layer):
                break

    ### compute the effective stride
    layers_with_strides = [[name for name in seq if hasattr(find_layer(model, name), 'strides')] for seq in sequences]
    layer_strides = [[find_layer(model, name).strides[0] for name in seq if hasattr(find_layer(model, name), 'strides')] for seq in sequences]
    effective_stride = {}
    for i,sequence in enumerate(sequences):
        for j,layer_name in enumerate(sequence):
            layer = find_layer(model, layer_name)
            if not include_stop_layer and is_stop_layer(layer):
                break
            if layer.name in layers_with_strides[i]:
                idx = layers_with_strides[i].index(layer.name)
                effective_stride[layer.name] = np.prod(layer_strides[i][:idx+1])
            elif j > 0:
                # a layer with no intrinsic stride has the same stride of the
                # last previous layer that has a stride
                prev_layers = [find_layer(model, name) for name in sequence[j-1::-1]]
                for prev_layer in prev_layers:
                    if prev_layer.name in effective_stride:
                        effective_stride[layer.name] = effective_stride[prev_layer.name]
                        break
            else:
                # the first layer
                effective_stride[layer.name] = 1
            if is_stop_layer(layer):
                break

    return effective_RF_size, effective_stride


def compute_correlations(model, X, fs, bands, effective_RF_size, effective_stride, filter_order=6, verbose=True):

    import sys
    def my_print(msg, fd=sys.stdout):
        fd.write(msg)
        fd.flush()

    from scipy.signal import butter, filtfilt, hilbert
    from tqdm import tqdm

    ## Filter the input in a series of bands and compute the signal envelope
    N_trials, N_samples = X.shape
    N_bands = len(bands)
    # filter the input in various frequency bands
    X_filt = np.zeros((N_bands, N_trials, N_samples))
    print('  Number of bands:', N_bands)
    print(' Number of trials:', N_trials)
    print('Number of samples:', N_samples)
    if verbose: my_print(f'Filtering the input in {N_bands} frequency bands... ')
    for i in range(N_bands):
        b,a = butter(filter_order//2, bands[i], 'bandpass', fs=fs)
        X_filt[i,:,:] = filtfilt(b, a, X)
    if verbose: print('done.')
    # compute the envelope of the filtered signal
    if verbose: my_print(f'Computing the envelope of the filtered signals... ')
    X_filt_envel = np.abs(hilbert(X_filt))
    if verbose: print('done.')

    ## Compute the outputs of the last layer of the model
    layer_name = model.layers[-1].name
    if verbose: my_print(f'Computing the output of layer {layer_name}... ')
    multi_Y = model(X)
    if verbose: print('done.')
    Y = multi_Y[-1].numpy() if isinstance(multi_Y, list) else multi_Y
    _, N_neurons, N_filters = Y.shape
    if verbose: print(f'Layer "{layer_name}" has {N_filters} filters, each with {N_neurons} neurons.')

    ## Compute the mean squared envelope for each receptive field
    RF_sz, RF_str = effective_RF_size[layer_name], effective_stride[layer_name]
    if verbose: print(f'The effective RF size and stride of layer "{layer_name}" are {RF_sz} and {RF_str} respectively.')
    mean_squared_envel = np.zeros((N_trials, N_bands, N_neurons))
    if verbose: my_print('Computing the mean squared envelope for each receptive field... ')
    for i in range(N_neurons):
        start, stop = i * RF_str, i * RF_str + RF_sz
        X_filt_envel_sub = X_filt_envel[:, :, start:stop]
        mean_squared_envel[:,:,i] = np.mean(X_filt_envel_sub ** 2, axis=2).T
    if verbose: print('done.')

    try:
        import ctypes
        libcorr = ctypes.CDLL(os.path.join('.', 'libcorr.so'))
        libcorr.pearsonr.argtypes = [ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double)]
        pointer = ctypes.POINTER(ctypes.c_double)
        R_pointer = pointer(ctypes.c_double(0.0))
        p_pointer = pointer(ctypes.c_double(0.0))
        def pearsonr(x, y):
            x = x.copy().astype(np.float64)
            y = y.copy()
            x_pointer = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            y_pointer = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            libcorr.pearsonr(x_pointer, y_pointer, x.size, R_pointer, p_pointer)
            return R_pointer[0], p_pointer[0]
    except:
        from scipy.stats import pearsonr
 
    ## For each frequency band, compute the correlation between mean squared envelope
    ## of the input (to each receptive field) and the output of each neuron in the layer
    R = np.zeros((N_trials, N_bands, N_filters))
    p = np.zeros((N_trials, N_bands, N_filters))
    if verbose: print('Computing the correlations tensor...')
    start = time()
    for i in tqdm(range(N_trials)):
        for j in range(N_bands):
            for k in range(N_filters):
                R[i,j,k], p[i,j,k] = pearsonr(Y[i,:,k], mean_squared_envel[i,j,:])
    stop = time()
    if verbose: print(f'Elapsed time: {(stop - start):.0f} sec.')
    return R, p

