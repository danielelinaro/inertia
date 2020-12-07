import os
import sys
import glob
from time import strftime, localtime
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_one_block(filename, varname, trial_dur = 60, verbose = False):
    """
    This function loads a single data file containing the results of the simulations for one value of inertia
    """
    data = np.load(filename)
    orig_n_trials, orig_n_samples = data[varname].shape
    dt = np.diff(data['time'][:2])[0]
    n_samples = int(trial_dur / dt)
    n_trials = int(orig_n_trials * orig_n_samples / n_samples)
    var = np.reshape(data[varname], [n_trials, n_samples], order='C')
    time = data['time'][:n_samples]
    if verbose:
        print('There are {} trials, each of which contains {} samples.'.\
              format(n_trials, n_samples))
    return tf.constant(time, dtype=tf.float32), \
           tf.constant(var, dtype=tf.float32), \
           tf.constant([float(data['inertia']) for _ in range(n_trials)], \
                       shape=(n_trials,1), dtype=tf.float32)


def build_network(depth_level = 1, learning_rate = 1e-4, dropout_coeff = None, loss_function = 'mae', optimizer = 'adam', full_output = False):
    """
    Builds the network
    
    The network topology used here is taken from the following paper:
   
    George, D., & Huerta, E. A. (2018).
    Deep neural networks to enable real-time multimessenger astrophysics.
    Physical Review D, 97(4), 044039. http://doi.org/10.1103/PhysRevD.97.044039
    """

    if loss_function.lower() == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError()
    elif loss_function.lower() == 'mape':
        loss = tf.keras.losses.MeanAbsolutePercentageError()
    else:
        print('Unknown loss function: {}.'.format(loss_function))
        return None

    if optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print('Unknown optimizer: {}.'.format(optimizer))
        return None    

    N_units = {}

    if depth_level == 1:
        N_units['conv'] = [16, 32, 64]
    elif depth_level == 2:
        N_units['conv'] = [64, 128, 256, 512]

    N_units['pooling'] = [4 for _ in range(len(N_units['conv']))]
    kernel_size = [5 for _ in range(len(N_units['conv']))]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((N,1), input_shape=(N,)),
    ])

    for N_conv,N_pooling,sz in zip(N_units['conv'], N_units['pooling'], kernel_size):
        model.add(tf.keras.layers.Conv1D(N_conv, sz, activation=None))
        model.add(tf.keras.layers.MaxPooling1D(N_pooling))
        model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Flatten())
    if depth_level == 2:
        model.add(tf.keras.layers.Dense(128, activation='relu'))
    
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    if dropout_coeff is not None:
        model.add(tf.keras.layers.Dropout(dropout_coeff))

    model.add(tf.keras.layers.Dense(y['train'].shape[1]))

    model.compile(optimizer=optimizer, loss=loss)

    if full_output:
        return model, N_units, kernel_size, optimizer, loss

    return model


def train_network(model, x, y, N_epochs, batch_size, steps_per_epoch = None, output_dir = '.', verbose = 1, full_output = False):
    ts = strftime('%Y%m%d-%H%M%S', localtime())
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    path = output_dir + '/' + ts
    checkpoint_path = path + '/checkpoints'
    os.makedirs(checkpoint_path)

    # create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + \
                                                     '/weights.{epoch:02d}-{val_loss:.2f}.h5',
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     monitor='val_loss',
                                                     verbose=0)

    if steps_per_epoch is None:
        N_batches = np.ceil(x['train'].shape[0] / batch_size)
        steps_per_epoch = np.max([N_batches, 100])

    history = model.fit(x['train'], y['train'], epochs=N_epochs, batch_size=batch_size,
                     steps_per_epoch=steps_per_epoch, validation_data=(x['val'], y['val']),
                     verbose=verbose, callbacks=[cp_callback])
    if full_output:
        return history, steps_per_epoch, path

    return history


if __name__ == '__main__':

    ### fix the seed of the random number generator, for reproducibility purposes
    use_good_seed = False

    with open('/dev/random', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')

    if use_good_seed:
        seed = 1057901520

    tf.random.set_seed(seed)

    print('Seed: {}'.format(seed))

    ### load the data
    x = {}
    y = {}
    data_folder = 'pan/npz_files/'
    sys.stdout.write('Loading data... ')
    sys.stdout.flush()
    for H in range(2,11):
        for i,cond in enumerate(('training', 'test', 'validation')):
            time, omega, inertia = load_one_block(data_folder + \
                                                  'ieee14_{}_set_H_{:.3f}.npz'.\
                                                  format(cond, H+i/3), \
                                                  'omega_coi', 60)
            key = cond[:5-i]
            try:
                x[key] = tf.concat([x[key], omega], axis=0)
                y[key] = tf.concat([y[key], inertia], axis=0)
            except:
                x[key] = omega
                y[key] = inertia
    sys.stdout.write('done.\n')
    N = x['train'].shape[1]

    ### normalize the data
    x_train_mean = np.mean(x['train'])
    x_train_std = np.std(x['train'])
    for key in x:
        x[key] = (x[key] - x_train_mean) / x_train_std

    with_dropout = False
    if with_dropout:
        dropout_coeff = 0.2
    else:
        dropout_coeff = None
    depth_level = 1
    learning_rate = 1e-4
    loss_function =  'MAE'
    model, N_units, kernel_size, optimizer, loss = build_network(depth_level, learning_rate, dropout_coeff, loss_function, full_output=True)
    model.summary()

    ### train the network
    N_epochs   = 3000
    batch_size = 128
    history, steps_per_epoch, output_path = train_network(model, x, y, N_epochs, batch_size, \
                                                          output_dir='inertia', full_output=True)
    checkpoint_path = output_path + '/checkpoints'
    
    ### find the best model based on the validation loss
    checkpoint_files = glob.glob(checkpoint_path + '/*.h5')
    val_loss = [float(file[:-3].split('-')[-1]) for file in checkpoint_files]
    best_checkpoint = checkpoint_files[np.argmin(val_loss)]
    best_model = tf.keras.models.load_model(best_checkpoint)

    ### compute the network prediction on the test set
    y_cnn = best_model.predict(x['test'])

    ### compute the mean absolute percentage error on the CNN prediction
    mape_cnn = tf.keras.losses.mean_absolute_percentage_error(tf.transpose(y['test']), \
                                                              tf.transpose(y_cnn)).numpy()[0]
    print('MAPE on CNN prediction ... {:.2f}%'.format(mape_cnn))

    parameters = {'N_samples': N, 'seed': seed, 'dropout_coeff': dropout_coeff,
                  'depth_level': depth_level, 'N_units': N_units, 'kernel_size': kernel_size,
                  'N_epochs': N_epochs, 'batch_size': batch_size, 'steps_per_epoch': steps_per_epoch,
                  'mape_cnn': mape_cnn, 'learning_rate': learning_rate, 'y_test': y['test'],
                  'y_cnn': y_cnn, 'loss_function': loss_function}

    best_model.save(output_path)
    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(history.history, open(output_path + '/history.pkl', 'wb'))

    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))
    ### plot the loss as a function of the epoch number
    epochs = np.r_[0 : len(history.history['loss'])] + 1
    ax1.plot(epochs, history.history['loss'], 'k', label='Training')
    ax1.plot(epochs, history.history['val_loss'], 'r', label='Validation')
    ax1.legend(loc='best')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ### plot the results obtained with the CNN
    limits = np.squeeze(y['train'].numpy()[[0,-1]])
    limits[1] += 1
    ax2.plot(limits, limits, 'g--')
    ax2.plot(y['test'], y_cnn, 'o', color=[1,.7,1], markersize=4, \
             markerfacecolor='w', markeredgewidth=1)
    for i in range(int(limits[0]), int(limits[1])):
        idx,_ = np.where(np.abs(y['test'] - (i + 1/3)) < 1e-3)
        m = np.mean(y_cnn[idx])
        s = np.std(y_cnn[idx])
        ax2.plot(i+1/3 + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
        ax2.plot(i+1/3, m, 'ms', markersize=8, markerfacecolor='w', markeredgewidth=2)
    ax2.set_title('CNN')
    ax2.set_xlabel('Expected value')
    ax2.set_ylabel('Predicted value')
    ax2.axis([1.8, limits[1], 0, limits[1]])
    plt.savefig(output_path + '/summary.pdf')

