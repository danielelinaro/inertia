
import os
import sys
import glob
import pickle
from comet_ml.api import API, APIExperiment
from comet_ml.query import Tag

if __name__ == '__main__':
    verbose = False
    if len(sys.argv) == 2:
        if sys.argv[1] in ('-v', '--verbose'):
            verbose = True
        else:
            print(f'{os.path.basename(sys.argv[0])}: unknown option: "{sys.argv[1]}"')

    expt_dir = os.path.join('experiments', 'neural_network')
    if not os.path.isdir(expt_dir):
        print(f'{expt_dir}: no such directory')
        sys.exit(1)

    EXPT_ID_LEN = 32
    Vds_only, Vqs_only = [], []
    for initial in '0123456789abcdef':
        expts = sorted([os.path.basename(d) for d in glob.glob(os.path.join(expt_dir, initial + '*')) \
                        if len(os.path.basename(d)) == EXPT_ID_LEN])
        if verbose: print(f'{len(expts)} experiments starting with "{initial}"')
        for expt in expts:
            pars_file = os.path.join(expt_dir, expt, 'parameters.pkl')
            if os.path.isfile(pars_file):
                var_names = pickle.load(open(pars_file, 'rb'))['var_names']
                if all(['Vd' in v for v in var_names]):
                    Vds_only.append(expt)
                elif all(['Vq' in v for v in var_names]):
                    Vqs_only.append(expt)
            elif verbose:
                print(f'{pars_file}: no such file')
    print(f'{len(Vds_only)} experiments use only Vd variables.')
    print(f'{len(Vqs_only)} experiments use only Vq variables.')

    #### Comet stuff
    def add_tag_to_expt(expt, tag, verbose=True):
        if verbose:
            sys.stdout.write(f"Adding tag '{tag}' to experiment '{expt.name}' [{expt.id[:6]}]... ")
            sys.stdout.flush()
        expt.add_tag(tag)
        if verbose:
            sys.stdout.write('done.\n')
    api = API(api_key = os.environ['COMET_API_KEY'])
    workspace = 'danielelinaro'
    project_name = 'inertia'
    query = Tag('IEEE39') & Tag('neural_network')
    sys.stdout.write(f'Querying Comet for experiments that match the following query: "{query}"... ')
    sys.stdout.flush()
    expts = api.query(workspace, project_name, query, archived=False)
    sys.stdout.write('done.\n')
    print(f'{len(expts)} experiments match the query.')
    for i,expt in enumerate(expts):
        ID = expt.id
        if ID in Vds_only or ID in Vqs_only:
            sys.stdout.write(f"[{i:3d}] Getting tags for experiment '{expt.name}' [{ID[:6]}]... ")
            sys.stdout.flush()
            tags = expt.get_tags()
            sys.stdout.write('done.\n')
            if ID in Vds_only and 'Vd' not in tags:
                add_tag_to_expt(expt, 'Vd')
            elif ID in Vqs_only and 'Vq' not in tags:
                add_tag_to_expt(expt, 'Vq')
