import json
import os
from pathlib import Path
from pathlib import PosixPath
from typing import List

import argparse




def make_parser() -> argparse.Namespace:
    """
    The function prepares argument parser for CLI application

    Returns:
        Namespace
    """


    parser = argparse.ArgumentParser(prog='GNN runner.')
    version = parser.add_argument('--version', action='store_true')
    size = parser.add_argument('-df','--df', action='store_true')
    
    
    subparsers = parser.add_subparsers(dest='command', required=False)
    
    mkproj = subparsers.add_parser('mkproj', help='Creates new project.')
    mkproj.add_argument('-n', '--name', type=str, help='Name of the project to create.')
    mkproj.add_argument('--train_path', type=str, help='Path to train mols in sdf format')
    mkproj.add_argument('--test_path', type=str, help='Path to test mols in sdf format')
    mkproj.add_argument('--property', type=str, help='Name of the property in SDF that contains property data')
    
    
    train_model = subparsers.add_parser('train', help='Train the prepared model')
    train_model.add_argument('-n', '--name', type=str, default=None, required=True)


    change_config = subparsers.add_parser('change', help='Train the prepared model')
    change_config.add_argument('-n', '--name', type=str, default=None, required=False)

    delete_prj = subparsers.add_parser('delete', help='Delete project.')
    delete_prj.add_argument('-n', '--name', type=str, default=None)
    
    args = parser.parse_args()

    return args