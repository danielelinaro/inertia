
import os
import signal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, callbacks
import tensorflow_addons as tfa

from .utils import print_msg, print_warning


__all__ = ['LEARNING_RATE', 'LearningRateCallback', 'make_preprocessing_pipeline_1D',
           'make_preprocessing_pipeline_2D', 'build_model', 'train_model', 'predict',
           'sigint_handler']

LEARNING_RATE = []

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


def make_preprocessing_pipeline_1D(input_layer, N_units, kernel_size, activation_fun, activation_loc, count=None):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function {activation_fun}')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    base_name = input_layer.name
    for n,(N_conv,N_pooling,sz) in enumerate(zip(N_units['conv'], N_units['pooling'], kernel_size)):
        conv_lyr_name = base_name + (f'_conv_{count}_{n+1}' if count is not None else f'_conv_{n+1}')
        activ_lyr_name = base_name + (f'_relu-{count}_{n+1}' if count is not None else f'_relu_{n+1}')
        pool_lyr_name = base_name + (f'_pool_{count}_{n+1}' if count is not None else f'_pool_{n+1}')
        try:
            L = layers.Conv1D(N_conv, sz, activation=None, name=conv_lyr_name)(L)
        except:
            L = layers.Conv1D(N_conv, sz, activation=None, name=conv_lyr_name)(input_layer)
        if activation_fun is not None:
            if activation_loc == 'after_conv':
                L = layers.ReLU(name=activ_lyr_name)(L)
                L = layers.MaxPooling1D(N_pooling,  name=pool_lyr_name)(L)
            else:
                L = layers.MaxPooling1D(N_pooling, name=pool_lyr_name)(L)
                L = layers.ReLU(name=activ_lyr_name)(L)
        else:
            L = layers.MaxPooling1D(N_pooling, name=pool_lyr_name)(L)
    return L


def make_preprocessing_pipeline_2D(input_layer, N_units, kernel_size, activation_fun, activation_loc, count=None):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function {activation_fun}')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    base_name = input_layer.name
    for n,(N_conv,N_pooling,sz) in enumerate(zip(N_units['conv'], N_units['pooling'], kernel_size)):
        conv_lyr_name = base_name + (f'_conv_{count}_{n+1}' if count is not None else f'_conv_{n+1}')
        activ_lyr_name = base_name + (f'_relu_{count}_{n+1}' if count is not None else f'_relu_{n+1}')
        pool_lyr_name = base_name + (f'_pool_{count}_{n+1}' if count is not None else f'_pool_{n+1}')
        try:
            L = layers.Conv2D(N_conv, [sz, 2], padding='same', activation=None, name=conv_lyr_name)(L)
        except:
            L = layers.Conv2D(N_conv, [sz, 2], padding='same', activation=None, name=conv_lyr_name)(input_layer)
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
                normalization_strategy, loss_fun_pars, optimizer_pars, lr_schedule_pars):
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

    loss_fun_name = loss_fun_pars['name'].lower()
    if loss_fun_name == 'mae':
        loss = losses.MeanAbsoluteError()
    elif loss_fun_name == 'mape':
        loss = losses.MeanAbsolutePercentageError()
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
    kernel_size = model_arch['kernel_size']
    activation_fun = model_arch['preproc_activation']
    activation_loc = model_arch['activation_loc']

    ### figure out how data should be normalized
    if normalization_strategy not in ('batch', 'layer', 'training_set'):
        raise ValueError('normalization_strategy must be one of "batch", "layer" or "trainin_set"')

    batch_norm = normalization_strategy.lower() == 'batch'
    normalization_layer = normalization_strategy.lower() == 'layer'

    def make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, activation_fun, activation_loc, count):
        if N_dims == 1:
            L = []
            for input_layer in input_layers:
                lyr = make_preprocessing_pipeline_1D(input_layer, N_units, kernel_size, \
                                                     activation_fun, activation_loc, count)
                L.append(lyr)
        else:
            L = make_preprocessing_pipeline_2D(input_layers[0], N_units, kernel_size, \
                                               activation_fun, activation_loc, count)
        return L

    def make_dense_stream(L, N_units, N_outputs, model_arch, count):
        for i,n in enumerate(N_units['dense']):
            L = layers.Dense(n, activation='relu', name=f'fc_{count}_{i+1}')(L)
        if model_arch['dropout_coeff'] > 0:
            L = layers.Dropout(model_arch['dropout_coeff'], name=f'dropout_{count}')(L)
        output = layers.Dense(N_outputs, name=f'predictions_{count}')(L)
        return output

    def make_full_stream(input_layers, N_dims, N_units, kernel_size, activation_fun, activation_loc, N_outputs, model_arch, count):
        L = make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, activation_fun, activation_loc, count)
        if isinstance(L, list):
            L = layers.concatenate(L, name=f'concat_{count}')
        L = layers.Flatten(name=f'flatten_{count}')(L)
        return make_dense_stream(L, N_units, N_outputs, model_arch, count)

    if type(streams_mode) not in (int, float):
        raise Exception('streams_mode must be an integer in the range [0,3]')
    if streams_mode not in (0, 1, 2, 3):
        raise ValueError('streams_mode must be in the range [0,3]')

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

    if streams_mode == 0:
        # just one stream
        outputs = [make_full_stream(input_layers, N_dims, N_units, kernel_size, activation_fun,
                                    activation_loc, N_outputs, model_arch, 1)]
    elif streams_mode == 1:
        # common preprocessing stream and then one stream of dense layers for each output
        L = make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, activation_fun, activation_loc, 1)
        if isinstance(L, list):
            L = layers.concatenate(L, name='concat_1')
        L = layers.Flatten(name='flatten_1')(L)
        outputs = [make_dense_stream(L, N_units, 1, model_arch, i+1) for i in range(N_outputs)]

    elif streams_mode == 2:
        # one preprocessing stream for each output and then one common stream of dense layers
        L = [make_preprocessing_stream(input_layers, N_dims, N_units, kernel_size, activation_fun,
                                       activation_loc, i+1) for i in range(N_outputs)]
        if isinstance(L[0], list):
            L = layers.concatenate([l for LL in L for l in LL], name='concat_1')
        else:
            L = layers.concatenate(L, name='concat_1')
        L = layers.Flatten(name='flatten_1')(L)
        outputs = make_dense_stream(L, N_units, N_outputs, model_arch, 1)

    elif streams_mode == 3:
        # one full stream for each output
        outputs = [make_full_stream(input_layers, N_dims, N_units, kernel_size, activation_fun,
                                    activation_loc, 1, model_arch, i+1) for i in range(N_outputs)]

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
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
    checkpoint_cb = callbacks.ModelCheckpoint(filepath = checkpoint_dir + \
                                              '/weights.{epoch:04d}-{val_loss:.6f}.h5',
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
                                                              min_lr = cb_pars['min_lr'])
                cbs.append(lr_scheduler_cb)
                print_msg('Added callback for reducing learning rate on plateaus.')
            else:
                raise Exception(f'Unknown callback: {cb_pars["name"]}')
    except:
        print_warning('Not adding callbacks.')

    if len(model.inputs) == 1:
        x_training = x['training']
        x_validation = x['validation']
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



