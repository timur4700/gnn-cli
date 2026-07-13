from typing import List
import os
from pathlib import Path, PosixPath
import json







def file_checker(path: str | PosixPath):
    exists = os.path.exists(path)
    return exists


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(path):

    with open(path) as f:
        file = json.load(f)

    return file


def looped_input(msg: str, data_type):
    print(msg)

    while True:
        user_input = input('INPUT: ')

        try:
            return data_type(user_input)

        except ValueError:
            pass

        print('Wrong input!')



def file_user_input(file_type: str, file_formats: List[str]) -> str:

    file_formats = ['.' + i for i in file_formats] 

    while True:
        user_input = input(f'Please, provide {file_type} path: ')

        if file_checker(user_input):
            if any(user_input.endswith(i) for i in file_formats):
                return user_input

        print(f"The file does not exists or wrong file formats (supported: {[', '.join(file_formats)]})!")



def calc_size(prj_name):

    if isinstance(prj_name, str):
        prj_name = Path(prj_name)

    total_size = 0

    for file in os.listdir(prj_name):

        size = os.path.getsize(prj_name / file)
        total_size += size

    return round(total_size / (1024**2), 2)