import os
import sys
import re
import glob
import json
import argparse as arg
from time import strftime, localtime
import pickle

import numpy as np
import matplotlib.pyplot as plt
from comet_ml import Experiment
import tensorflow as tf
import tensorflow_addons as tfa
from deep_utils import *

import colorama as cm
print_error   = lambda msg: print(f'{cm.Fore.RED}'    + msg + f'{cm.Style.RESET_ALL}')
print_warning = lambda msg: print(f'{cm.Fore.YELLOW}' + msg + f'{cm.Style.RESET_ALL}')
print_msg     = lambda msg: print(f'{cm.Fore.GREEN}'  + msg + f'{cm.Style.RESET_ALL}')



def build_model(N_samples, model_arch, loss_fun_pars, optimizer_pars):
    """
    Builds and compiles the model
    
    The network topology used here is taken from the following paper:
   
    George, D., & Huerta, E. A. (2018).
    Deep neural networks to enable real-time multimessenger astrophysics.
    Physical Review D, 97(4), 044039. http://doi.org/10.1103/PhysRevD.97.044039
    """

    loss_fun_name = loss_fun_pars['name'].lower()
    if loss_fun_name == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError()
    elif loss_fun_name == 'mape':
        loss = tf.keras.losses.MeanAbsolutePercentageError()
    else:
        raise Exception('Unknown loss function: {}.'.format(loss_function))

    learning_rate = optimizer_pars['learning_rate']
    optimizer_name = optimizer_pars['name'].lower()
    if optimizer_name == 'sgd':
        momentum = optimizer_pars['momentum'] if 'momentum' in optimizer_pars else 0.
        nesterov = optimizer_pars['nesterov'] if 'nesterov' in optimizer_pars else False
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum, nesterov)
    elif optimizer_name in ('adam', 'adamax', 'nadam'):
        beta_1 = optimizer_pars['beta_1'] if 'beta_1' in optimizer_pars else 0.9
        beta_2 = optimizer_pars['beta_2'] if 'beta_2' in optimizer_pars else 0.999
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2)
        elif optimizer_name == 'adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate, beta_1, beta_2)
        else:
            optimizer = tf.keras.optimizers.Nadam(learning_rate, beta_1, beta_2)
    elif optimizer_name == 'adagrad':
        initial_accumulator_value = optimizer_pars['initial_accumulator_value'] if \
                                    'initial_accumulator_value' in optimizer_pars else 0.1
        optimizer = tf.keras.optimizers.Adagrad(learning_rate, initial_accumulator_value)
    elif optimizer_name == 'adadelta':
        rho = optimizer_pars['rho'] if 'rho' in optimizer_pars else 0.95
        optimizer = tf.keras.optimizers.Adadelta(learning_rate, rho)
    else:
        raise Exception('Unknown optimizer: {}.'.format(optimizer_name))

    N_units = model_arch['N_units']
    kernel_size = model_arch['kernel_size']

    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((N_samples,1), input_shape=(N_samples,)),
    ])

    for N_conv,N_pooling,sz in zip(N_units['conv'], N_units['pooling'], kernel_size):
        model.add(tf.keras.layers.Conv1D(N_conv, sz, activation=None))
        model.add(tf.keras.layers.MaxPooling1D(N_pooling))
        model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Flatten())
    for n in N_units['dense']:
        model.add(tf.keras.layers.Dense(n, activation='relu'))

    if model_arch['dropout_coeff'] > 0:
        model.add(tf.keras.layers.Dropout(model_arch['dropout_coeff']))

    model.add(tf.keras.layers.Dense(y['training'].shape[1]))

    model.compile(optimizer=optimizer, loss=loss)

    return model, optimizer, loss



def train_model(model, x, y,
                N_epochs,
                batch_size,
                steps_per_epoch,
                output_dir,
                early_stopping_patience = None,
                learning_rate_schedule = None,
                verbose = 1):

    checkpoint_dir = output_dir + '/checkpoints'
    os.makedirs(checkpoint_dir)

    # create a callback that saves the model's weights
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_dir + \
                                                       '/weights.{epoch:02d}-{val_loss:.2f}.h5',
                                                       save_weights_only = False,
                                                       save_best_only = True,
                                                       monitor = 'val_loss',
                                                       verbose = verbose)
    cbs = [checkpoint_cb]

    if early_stopping_patience is not None:
        # create a callback that will stop the optimization if there is no improvement
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                         patience = early_stopping_patience,
                                                         verbose = verbose,
                                                         mode = 'min')

        cbs.append(early_stop_cb)
        print_msg('Added a callback for early stopping.')
    else:
        print_warning('Not adding a callback for early stopping.')

    try:

        if learning_rate_schedule['name'] == 'cyclical':
            # According to [1], "experiments show that it often is good to set stepsize equal to
            # 2 − 10 times the number of iterations in an epoch".
            #
            # [1] Smith, L.N., 2017, March.
            #     Cyclical learning rates for training neural networks.
            #     In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 464-472). IEEE.
            #
            schedule = tfa.optimizers.Triangular2CyclicalLearningRate(
                initial_learning_rate = learning_rate_schedule['initial_learning_rate'],
                maximal_learning_rate = learning_rate_schedule['max_learning_rate'],
                step_size = x['training'].shape[0] // batch_size * learning_rate_schedule['factor'])
            lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(schedule, verbose)
            cbs.append(lr_scheduler_cb)
            print_msg('Added a callback for cyclical learning rate scheduling.')

        elif learning_rate_schedule['name'] == 'reduce_on_plateau':
            optimizer_lr = model.optimizer.learning_rate.numpy()
            factor = learning_rate_schedule['factor']
            patience = learning_rate_schedule['patience']
            cooldown = learning_rate_schedule['cooldown'] if 'cooldown' \
                       in learning_rate_schedule else patience / 10
            min_lr = learning_rate_schedule['min_learning_rate'] if 'min_learning_rate' \
                     in learning_rate_schedule else optimizer_lr / 1000
            reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                                factor = factor,
                                                                patience = patience,
                                                                verbose = verbose,
                                                                mode = 'min',
                                                                cooldown = cooldown,
                                                                min_lr = min_lr)
            cbs.append(reduce_lr_cb)
            print_msg('Added a callback for reducing learning rate on plateaus.')

        else:
            raise Exception('Unknown learning rate scheduling policy: {}'.format(learning_rate_schedule['name']))

    except:
        print_warning('Not adding a callback for learning rate scheduling.')

    return model.fit(x['training'], y['training'], epochs=N_epochs, batch_size=batch_size,
                     steps_per_epoch=steps_per_epoch, validation_data=(x['validation'], y['validation']),
                     verbose=verbose, callbacks=cbs)



if __name__ == '__main__':

    progname = os.path.basename(sys.argv[0])
    
    parser = arg.ArgumentParser(description = 'Train a network to estimate inertia',
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-o', '--output-dir',  default='experiments',  type=str, help='output directory')
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print_error('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    with open('/dev/random', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    tf.random.set_seed(seed)
    print_msg('Seed: {}'.format(seed))

    log_to_comet = not args.no_comet

    if log_to_comet:
        ### create a CometML experiment
        experiment = Experiment(
            api_key = os.environ['COMET_ML_API_KEY'],
            project_name = 'inertia',
            workspace = 'danielelinaro'
        )
        experiment_key = experiment.get_key()
    else:
        experiment_key = strftime('%Y%m%d-%H%M%S', localtime())

    ### load the data
    data_folder = config['data_dir']
    if not os.path.isdir(data_folder):
        print_error('{}: {}: no such directory.'.format(progname, data_folder))
        sys.exit(1)

    inertia = {}
    for key in ('training', 'test', 'validation'):
        inertia[key] = np.sort([float(re.findall('[0-9]+\.[0-9]*', f)[-1]) for f in glob.glob(data_folder + '/*' + key + '*.npz')])

    time, x, y = load_data(data_folder, inertia, config['var_name'])
    N_training_traces, N_samples = x['training'].shape

    ### normalize the data
    x_train_mean = np.mean(x['training'])
    x_train_std = np.std(x['training'])
    for key in x:
        x[key] = (x[key] - x_train_mean) / x_train_std

    model, optimizer, loss = build_model(N_samples,
                                         config['model_arch'],
                                         config['loss_function'],
                                         config['optimizer'])
    model.summary()

    ### train the network
    parameters = config.copy()
    parameters['seed'] = seed
    parameters['N_training_traces'] = N_training_traces
    parameters['N_samples'] = N_samples
    parameters['x_train_mean'] = x_train_mean
    parameters['x_train_std'] = x_train_std
    
    try:
        es_patience = config['early_stopping_patience']
    except:
        es_patience = None
        parameters['early_stopping_patience'] = None

    try:
        lr_schedule = config['learning_rate_schedule']
    except:
        lr_schedule = None
        parameters['learning_rate_schedule'] = None

    N_epochs = config['N_epochs']
    batch_size = config['batch_size']
    N_batches = np.ceil(N_training_traces / batch_size)
    parameters['N_batches'] = N_batches
    steps_per_epoch = np.max([N_batches, 100])
    parameters['steps_per_epoch'] = steps_per_epoch
    output_path = args.output_dir + '/' + experiment_key
    parameters['output_path'] = output_path

    if log_to_comet:
        experiment.log_parameters(parameters)

    history = train_model(model, x, y,
                          N_epochs,
                          batch_size,
                          steps_per_epoch,
                          output_path,
                          es_patience,
                          lr_schedule,
                          verbose = 1)

    checkpoint_path = output_path + '/checkpoints'
    
    ### find the best model based on the validation loss
    checkpoint_files = glob.glob(checkpoint_path + '/*.h5')
    val_loss = [float(file[:-3].split('-')[-1]) for file in checkpoint_files]
    best_checkpoint = checkpoint_files[np.argmin(val_loss)]
    best_model = tf.keras.models.load_model(best_checkpoint)

    ### compute the network prediction on the test set
    y_prediction = np.squeeze(best_model.predict(x['test']))

    ### compute the mean absolute percentage error on the CNN prediction
    y_test = np.squeeze(y['test'].numpy())
    mape_prediction = tf.keras.losses.mean_absolute_percentage_error(y_test, y_prediction).numpy()
    print_msg('MAPE on CNN prediction ... {:.2f}%'.format(mape_prediction))
    test_results = {'y_test': y_test, 'y_prediction': y_prediction, 'mape_prediction': mape_prediction}
    
    best_model.save(output_path)
    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(test_results, open(output_path + '/test_results.pkl', 'wb'))
    pickle.dump(history.history, open(output_path + '/history.pkl', 'wb'))

    if log_to_comet:
        experiment.log_metric('mape_prediction', mape_prediction)
        experiment.log_model('best_model', output_path + '/saved_model.pb')

    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))
    ### plot the loss as a function of the epoch number
    epochs = np.r_[0 : len(history.history['loss'])] + 1
    ax1.plot(epochs, history.history['loss'], 'k', label='Training')
    ax1.plot(epochs, history.history['val_loss'], 'r', label='Validation')
    ax1.legend(loc='best')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ### plot the results obtained with the CNN
    limits = np.squeeze(y['training'].numpy()[[0,-1]])
    limits[1] += 1
    ax2.plot(limits, limits, 'g--')
    ax2.plot(y['test'], y_prediction, 'o', color=[1,.7,1], markersize=4, \
             markerfacecolor='w', markeredgewidth=1)
    for i in range(int(limits[0]), int(limits[1])):
        idx,_ = np.where(np.abs(y['test'] - (i + 1/3)) < 1e-3)
        m = np.mean(y_prediction[idx])
        s = np.std(y_prediction[idx])
        ax2.plot(i+1/3 + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
        ax2.plot(i+1/3, m, 'ms', markersize=8, markerfacecolor='w', markeredgewidth=2)
    ax2.set_title('CNN')
    ax2.set_xlabel('Expected value')
    ax2.set_ylabel('Predicted value')
    ax2.axis([1.8, limits[1], 0, limits[1]])
    plt.savefig(output_path + '/summary.pdf')

