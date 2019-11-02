import argparse
from pathlib import Path

import mlh_challenge.data
import mlh_challenge.run
from mlh_challenge import DATA_DIR, MODELS_DIR, OUT_DIR


def parse_cli():
    def is_file(filename: str, reldir: Path = None):
        filepath = Path(filename)
        if filepath.is_file():
            return filepath
        if not reldir or not reldir.joinpath(filepath).is_file():
            raise argparse.ArgumentTypeError(f"Can't find file {filename}.")
        return reldir.joinpath(filepath)

    help_formatter = argparse.ArgumentDefaultsHelpFormatter

    p = argparse.ArgumentParser(description='Make datasets',
                                formatter_class=help_formatter)
    p.set_defaults(handler=None)

    # Subcommands
    sp = p.add_subparsers(help='Available actions', dest='action')

    # Training
    # TODO: You can add extra args here which will be passed to the
    #  training() function in the run.py module.
    sp_train = sp.add_parser('train', help='Run training',
                             formatter_class=help_formatter)
    sp_train.set_defaults(handler=mlh_challenge.run.training)
    sp_train.add_argument('--data-file', required=False,
                          default=DATA_DIR.joinpath('mlh-train.npz'),
                          type=is_file, help='Input data file')
    sp_train.add_argument('--save-model', required=False,
                          default=MODELS_DIR.joinpath('model.pkl'),
                          help='Name path to model file to save.'
                               'If None or empty, model will not be saved.')

    # Inference
    sp_infer = sp.add_parser('infer', help='Run inference',
                             formatter_class=help_formatter)
    sp_infer.set_defaults(handler=mlh_challenge.run.inference)
    sp_infer.add_argument('--data-file', required=False,
                          default=DATA_DIR.joinpath('mlh-test.npz'),
                          type=is_file, help='Input data file')
    sp_infer.add_argument('--out-file', required=False,
                          default=OUT_DIR.joinpath('results.csv'),
                          type=str, help='Output file path, for writing '
                                         'classification results. Can be '
                                         'a .csv/tsv/xls/xlsx file.')
    sp_infer.add_argument('--load-model', required=True, type=is_file,
                          help='Path to model file to load')

    parsed = p.parse_args()
    if not parsed.action:
        p.error("Please specify an action")

    return parsed


if __name__ == '__main__':
    parsed_args = parse_cli()
    parsed_args.handler(**vars(parsed_args))
