## -*- coding: utf-8 -*-
"""ND2Figures
Version 1.1
Dillon Brown, 22March2022
"""

import argparse, json
from pathlib import Path
from app import analyze_folder, get_user_path
from app.config import ConfigContainer as cc

if __name__ == '__main__':
    print('test')

    parser = argparse.ArgumentParser()
    parser.add_argument('--clear',
                        help="Clear previous exports",
                        default=False,
                        action='store_true')

    parser.add_argument('--show',
                        help="show plots",
                        default=False,
                        action='store_true')

    parser.add_argument('--default',
                        help="use default path",
                        default=False,
                        action='store_true')
    parser.add_argument('--repeat',
                        help="use same path as previous",
                        default=False,
                        action='store_true')
    parser.add_argument('--no-fig',
                        help="don't create png figure",
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--nooffset',
        help="Don't offset by auto-calculated background intensity",
        default=False,
        action='store_true')
    kwargs = vars(parser.parse_args())
    directory = cc.ihc_dir
    cc.create_fig = not kwargs['no_fig']

    if kwargs["repeat"]:
        with open(cc.LASTDIR, mode='r') as f:
            val = json.load(f)
            directory = Path(val)
    elif not kwargs["default"]:
        directory = get_user_path()

    with open(cc.LASTDIR, mode='w') as f:
        json.dump(directory.as_posix(), f)
    analyze_folder(directory,
                   clear=kwargs['clear'],
                   groupby=None,
                   identify=None,
                   disp=kwargs['show'],
                   offset=not kwargs['nooffset'])
