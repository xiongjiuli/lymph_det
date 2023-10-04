"""Helper functions for input/output."""

import os
import json
import logging
import pickle
from pathlib import Path
import sys
import socket
import subprocess

import numpy as np
import torch
import yaml
import SimpleITK as sitk

PATH_TO_CONFIG = Path("./config/")


def get_config(config_name):
    """Loads a .yaml file from ./config corresponding to the name arg.

    Args:
        config_name: A string referring to the .yaml file to load.

    Returns:
        A container including the information of the referred .yaml file and information
        regarding the dataset, if specified in the referred .yaml file.
    """
    with open(PATH_TO_CONFIG / (config_name + '.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)

    return config

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=3)


def get_meta_data():
    meta_data = {}
    meta_data['git_commit_hash'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    meta_data['python_version'] = sys.version.splitlines()[0]
    meta_data['gcc_version'] = sys.version.splitlines()[1]
    meta_data['pytorch_version'] = torch.__version__
    meta_data['host_name'] = socket.gethostname()

    return meta_data
