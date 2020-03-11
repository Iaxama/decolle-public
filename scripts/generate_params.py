import yaml
import os
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter file generator for grid search')
    parser.add_argument('--param_name', '-p', dest='param_name', type=str, required=True, action='append',
                        help='Name of parameter to change as it appears in default_params_file')
    parser.add_argument('--values_list', '-v', dest='values_list', required=True, nargs='+', action='append',
                        help='Set of values to test for the specified parameter. By default parsed as int. '
                             'If float add decimal (i.e -v 2.0). If list type comma separated entries in turn separated by space.'
                             'Single element list must have a comma at the end (i.e -v 1,2,3 4, 5,6')
    parser.add_argument('--default_params_file', type=str, default='params.yml',
                        help='File containing all of the default parameters and the specific parameter to test')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    out_dir = 'params_to_test'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        [os.remove(x) for x in glob.glob(os.path.join(out_dir, '*'))]

    param_list = [args.default_params_file]

    for param_name, values in zip(args.param_name, args.values_list):
        for param_file in param_list:
            with open(param_file, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            generate_param_test(out_dir, params, param_name, values)
        param_list = [os.path.join(out_dir, x) for x in os.listdir(out_dir)]
        if not param_list:
            param_list = [args.default_params_file]


def generate_param_test(out_dir, params, param_name, values_list):
    t = None
    subt = None
    for p in values_list:
        # Parsing values
        params_out = params.copy()
        p_split = p.split(',')
        if len(p_split) > 1 or isinstance(params_out[param_name], list):
            if not p_split[-1]:
                del p_split[-1]
            try:
                new_value = [int(e) for e in p_split]
            except ValueError:
                new_value = [float(e) for e in p_split]

        else:
            try:
                new_value = int(p_split[0])
            except ValueError:
                new_value = float(p_split[0])

        if params_out[param_name] == new_value:
            continue  # Avoid duplicates

        params_out[param_name] = new_value
        try:
            last_idx = int(os.path.splitext(sorted(os.listdir(out_dir))[-1])[0])
        except IndexError:
            last_idx = 0
        with open(os.path.join(out_dir, '{:05d}.yml'.format(last_idx + 1)), 'w') as outfile:
            yaml.dump(params_out, outfile)

        # Checking type consistency
        warn = False

        if isinstance(params_out[param_name], list):
            if subt is not None:
                if not isinstance(params_out[param_name][0], subt):
                    warn = True
            subt = type(params_out[param_name][0])

        if t is not None:
            if not isinstance(params_out[param_name], t):
                warn = True
        t = type(params_out[param_name])

        if warn:
            print('WARNING! Check type consistency')


if __name__ == '__main__':
    main()


