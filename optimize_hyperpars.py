
import os
import re
import sys
import json
import pickle
from comet_ml import Optimizer


prog_name = os.path.basename(sys.argv[0])


def usage():
    print(f'usage: {prog_name} [<options>] <config_file>')
    print('')
    print('   --max-cores     maximum number of cores to be used by Keras')
    print('    -h, --help     print this help message and exit')
    print('')


if __name__ == '__main__':

    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        usage()
        sys.exit(0)

    i = 1
    n_args = len(sys.argv)
    training_config_file = os.path.join(os.path.sep, 'tmp', 'training_config.json')
    args = []
    
    while i < n_args:
        arg = sys.argv[i]
        if arg == '--max-cores':
            args += ['--max-cores', sys.argv[i+1]]
            i += 1
        else:
            break
        i += 1

    if i == n_args:
        usage()
        sys.exit(1)

    args.append(training_config_file)

    config_file = sys.argv[i]
    if not os.path.isfile(config_file):
        print(f'{prog_name}: {config_file}: no such file')
        sys.exit(2)

    # the configuration for this script
    config = json.load(open(config_file, 'r'))

    # the configuration for the model training script
    training_config = json.load(open(config['training_config_file'], 'r'))
    N_conv_layers = len(training_config['model_arch']['kernel_size'])

    optimizer_config_keys = 'algorithm', 'spec', 'parameters', 'name', 'trials'
    optimizer_config = {}
    for key in config:
        if key in optimizer_config_keys:
            optimizer_config[key] = config[key]
    optimizer = Optimizer(optimizer_config, api_key=os.environ['COMET_API_KEY'])

    experiment_ids = []
    kernel_size = []
    loss, val_loss = [], []
    import train_network
    for experiment in optimizer.get_experiments(project_name='inertia', workspace='danielelinaro'):

        kernel_size.append(experiment.get_parameter('kernel_size'))
        training_config['model_arch']['kernel_size'] = [kernel_size[-1] for _ in range(N_conv_layers)]
        json.dump(training_config, open(training_config_file, 'w'), indent=4)
        output_path = train_network.main(progname='train_network.py',
                                         args=args)
        history = pickle.load(open(os.path.join(output_path, 'history.pkl'), 'rb'))
        loss.append(min(history['loss']))
        val_loss.append(min(history['val_loss']))
        experiment_ids.append(experiment.id)
        experiment.log_metric('val_loss', val_loss[-1])

    data = {'experiment_ids': experiment_ids,
            'kernel_size': kernel_size,
            'loss': loss,
            'val_loss': val_loss,
            'config': config,
            'training_config': training_config,
            'optimizer_status': optimizer.status()}

    optimizer_id = re.findall('[a-z0-9]+', optimizer.get_id())[0]
    output_path = os.path.join('experiments',
                               'neural_network',
                               'hyperparameters_optimization',
                               optimizer_id)
    os.mkdir(output_path)
    pickle.dump(data, open(os.path.join(output_path, 'optimization_results.pkl'), 'wb'))

