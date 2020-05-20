import argparse
import yaml
# TODO: Unclear if this is needed
# from utils import dotdict

def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='Get config values'
    )
    # jbinfo args
    parser.add_argument(
        '-i', '--in-config',
        required=True,
        help='Path fo config'
    )
    parser.add_argument(
        '-g', '--get',
        help='Get value of field'
    )
    return parser.parse_args()

if __name__ == '__main__':

    # ARGUMENT HANDLING
    args = argument_parsing()

    # Read config from yaml file.
    with open(args.in_config) as fid:
        config = yaml.safe_load(fid)

    # print parameter on the command line
    if args.get and args.get in config:
        print(config[args.get])
