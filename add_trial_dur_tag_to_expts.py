
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
            if os.path.isfile(pars_file) and not any(['trial_dur' in tag for tag in expt.get_tags()]):
                pars = pickle.load(open(pars_file, 'rb'))
                if 'trial_duration' in pars:
                    trial_dur = pars['trial_duration']
                elif 'N_samples' in pars:
                    srate = 10
                    trial_dur = pars['N_samples'] / srate
                else:
                    import ipdb
                    ipdb.set_trace()
                expt.add_tag(f'trial_dur_{trial_dur:.0f}')
