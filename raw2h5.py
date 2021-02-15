
import os
import sys
import argparse as arg
import numpy as np
import tables
from scipy.interpolate import interp1d

from pypan.utils import *
from build_data import generator_ids

progname = os.path.basename(sys.argv[0])


if __name__ == '__main__':
    parser = arg.ArgumentParser(description = 'Extract data from a PAN binary file', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('file', type=str, action='store', help='Data file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    parser.add_argument('-a', '--append', action='store_true', help='append to existing file')
    parser.add_argument('-f', '--force', action='store_true', \
                        help='force overwrite of output file (ignored if -o is provided)')
    parser.add_argument('-v', '--variables', default=None, type=str, help='variables to save')
    parser.add_argument('-V', '--variable-names', default=None, type=str, help='variable names to use')
    parser.add_argument('-l', '--list', action='store_true', help='list variables contained in file')
    parser.add_argument('-q', '--quiet', action='store_true', help='do not print anything to terminal')
    parser.add_argument('-P', '--power', action='store_true', help='save power related variables')
    parser.add_argument('--frand', default=10., type=float, help='frequency of random numbers')
    args = parser.parse_args(args=sys.argv[1:])

    in_file = args.file
    if not os.path.isfile(in_file):
        print('{}: {}: no such file.'.format(progname, in_file))
        sys.exit(1)

    if args.list:
        print('\nVariables contained in {}:\n'.format(in_file))
        var_names, var_units, _ = get_vars_list(in_file)
        max_len_var_names = np.max(list(map(len, var_names))) + 1
        max_len_var_units = np.max(list(map(len, var_units))) + 1
        header = '{{:^{}s}}   {{:^{}s}}'.format(max_len_var_names, max_len_var_units)
        fmt = '{{:{}s}}   {{:{}s}}'.format(max_len_var_names, max_len_var_units)
        print(header.format('NAME', 'UNITS'))
        for name,units in zip(var_names,var_units):
            print(fmt.format(name, units))
        print('')
        sys.exit(0)

    if args.output is not None:
        out_file = args.output
    else:
        folder,filename = os.path.split(in_file)
        out_file = folder + '/' + os.path.splitext(filename)[0] + '.h5'

    if args.power and not os.path.isfile(out_file):
        print('{}: {}: file does not exist but required to extract time vector.'.format(progname, out_file))
        print('First run {} on an @noise file and then on the @devvars file.'.format(progname))
        sys.exit(2)
    elif not args.power and os.path.isfile(out_file) and args.output is None and not args.force:
        print('{}: {}: file exists. Use -f to overwrite.'.format(progname, out_file))
        sys.exit(3)

    if args.power:
        var_names = ['DevTime']
        for gen_id in generator_ids:
            for lbl in 'p','q':
                var_names.append(f'G{gen_id}:{lbl}e')
    elif args.variables is None:
        var_names,_,_ = get_vars_list(in_file)
    else:
        var_names = args.variables.split(',')

    if not args.quiet:
        sys.stdout.write('Loading data from {}... '.format(in_file))
        sys.stdout.flush()
    data = load_vars(in_file, var_names, mode='chunk')
    if not args.quiet:
        sys.stdout.write('done.\n')
    if args.power:
        dt = 1 / args.frand
        dev_time = data['DevTime']
        fid = tables.open_file(args.output, 'r')
        time = fid.root.time.read()
        fid.close()
        try:
            idx = np.array([np.where(dev_time == tt)[0][0] for tt in time])
            is_subset = True
        except:
            is_subset = False

    if not args.quiet:
        sys.stdout.write('Saving data to {}... '.format(out_file))
        sys.stdout.flush()

    if args.variable_names is None:
        conversion = {
            'omega01': 'omega_G1',
            'omega02': 'omega_G2',
            'G3:omega': 'omega_G3',
            'G6:omega': 'omega_G6',
            'G8:omega': 'omega_G8'
        }
        for gen_id in generator_ids:
            for lbl in 'p','q':
                conversion[f'G{gen_id}:{lbl}e'] = f'{lbl.upper()}e_G{gen_id}'
    else:
        conversion = {k: v for k, v in zip(var_names, args.variable_names.split(','))}

    if args.append:
        mode = 'a'
    else:
        mode = 'w'

    class Parameters (tables.IsDescription):
        frand = tables.Float64Col()

    compression_filter = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Float64Atom()

    fid = tables.open_file(args.output, mode, filters=compression_filter)

    try:
        tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
        params = tbl.row
        params['frand'] = args.frand
        params.append()
    except:
        pass

    for k,v in data.items():
        try:
            key = conversion[k]
        except:
            key = k
        if args.power and key != 'DevTime':
            if is_subset:
                fid.create_array(fid.root, key, v[idx])
            else:
                f = interp1d(dev_time, v)
                fid.create_array(fid.root, key, f(time))
        elif not args.power:
            if 'omegael' in key:
                offset = 1.0
            else:
                offset = 0.0
            fid.create_array(fid.root, key, v + offset)
    fid.close()

    if not args.quiet:
        sys.stdout.write('done.\n')
    
