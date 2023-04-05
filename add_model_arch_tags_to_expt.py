
import os
import sys
import glob
import pickle
from comet_ml.api import API, APIExperiment
from tqdm import tqdm

if __name__ == '__main__':
    expt_dir = os.path.join('experiments', 'neural_network')
    if not os.path.isdir(expt_dir):
        print(f'{expt_dir}: no such directory')
        sys.exit(1)

    api = API(api_key = os.environ['COMET_API_KEY'])
    workspace = 'danielelinaro'
    project_name = 'inertia'

    EXPT_ID_LEN = 32
    for initial in '0123456789abcdef':
        expt_keys = sorted([os.path.basename(d) for d in glob.glob(os.path.join(expt_dir, initial + '*')) \
                            if len(os.path.basename(d)) == EXPT_ID_LEN])
        for expt_key in tqdm(expt_keys):
            expt = api.get_experiment(workspace, project_name, expt_key)
            if expt is None:
                continue
            pars_file = os.path.join(expt_dir, expt_key, 'parameters.pkl')

            if os.path.isfile(pars_file):
                pars = pickle.load(open(pars_file, 'rb'))
                tags = expt.get_tags()
                if not any(['N_conv_units_' in tag for tag in tags]):
                    expt.add_tag('N_conv_units_' + '_'.join(map(str, pars['model_arch']['N_units']['conv'])))
                if not any(['N_pool_units_' in tag for tag in tags]):
                    expt.add_tag('N_pool_units_' + '_'.join(map(str, pars['model_arch']['N_units']['pooling'])))
                if not any(['N_dense_units_' in tag for tag in tags]):
                    expt.add_tag('N_dense_units_' + '_'.join(map(str, pars['model_arch']['N_units']['dense'])))
                if not any(['kernel_sizes_' in tag for tag in tags]):
                    expt.add_tag('kernel_sizes_' + '_'.join(map(str, pars['model_arch']['kernel_size'])))
                if not any(['kernel_strides_' in tag for tag in tags]):
                    if 'kernel_stride' in pars['model_arch']:
                        expt.add_tag('kernel_strides_' + '_'.join(map(str, pars['model_arch']['kernel_stride'])))
                    else:
                        expt.add_tag('kernel_strides_' + '_'.join(['1' for _ in range(len(pars['model_arch']['kernel_size']))]))
