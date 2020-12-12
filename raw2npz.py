
import os
import sys
import argparse as arg
import numpy as np

progname = os.path.basename(sys.argv[0])


if __name__ == '__main__':
    parser = arg.ArgumentParser(description = 'Extract data from a PAN binary file', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('file', type=str, action='store', help='Data file')
    parser.add_argument('-o', '--output',  default=None, type=str, help='output file name')
    parser.add_argument('-f', '--force', action='store_true', \
                        help='force overwrite of output file (ignored if -o is provided)')
    parser.add_argument('-v', '--variables', default=None, type=str, help='variables to save')
    parser.add_argument('-l', '--list', action='store_true', help='list variables contained in file')
    parser.add_argument('-q', '--quiet', action='store_true', help='do not print anything to terminal')
    args = parser.parse_args(args=sys.argv[1:])

    from pypan.utils import *

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
        out_file = folder + '/' + os.path.splitext(filename)[0] + '.npz'

    if os.path.isfile(out_file) and args.output is None and not args.force:
        print('{}: {}: file exists. Use -f to overwrite.'.format(progname, out_file))
        sys.exit(2)

    if args.variables is None:
        var_names,_,_ = get_vars_list(in_file)
    else:
        var_names = args.variables.split(',')

    if not args.quiet:
        sys.stdout.write('Loading data from {}... '.format(in_file))
        sys.stdout.flush()
    data = load_vars(in_file, var_names, mode='chunk')
    if not args.quiet:
        sys.stdout.write('done.\n')

    if not args.quiet:
        sys.stdout.write('Saving data to {}... '.format(out_file))
        sys.stdout.flush()
    np.savez_compressed(out_file, **data)    
    if not args.quiet:
        sys.stdout.write('done.\n')
    
