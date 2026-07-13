from rich.table import Table
from rich.console import Console

from cli import interactive
from utils import func
from dataclasses import fields, Field
from train.config import TrainerConfig

from proj.func import Configs

import numpy as np
def print_trainer_config(config,
                       title: str='Trainer Configuration'):
    
    console = Console()
    model_table = Table(title=title)
    model_table.add_column('Parameter')
    model_table.add_column('Current Value')


    for field in fields(config):
        name = field.name
        current = getattr(config, name)
        editable = field.metadata['editable']

        if not editable:
            continue


        model_table.add_row(name, str(current))    
    console.print(model_table)



def change_train_config(config):

    choices = [field.name for field in fields(config) if field.metadata['editable']] + ['Confirm']
 
    while True:

        print_trainer_config(config)
        parameter = interactive.query('Choose model parameter', choices, False)

        if parameter == 'Confirm':
            break


        field: Field = config.__dataclass_fields__[parameter]

        if 'choices' in field.metadata:
            parameter_value = interactive.query('Choose parameter', field.metadata['choices'], False)

        else:

            min_value, max_value = field.metadata['min'], field.metadata['max']
            parameter_value = func.looped_input('Input value: ', field.type)

            if 'min' in field.metadata and 'max' in field.metadata:

                min_value, max_value = field.metadata['min'], field.metadata['max']

                if parameter_value < min_value:
                    print(f'{parameter} cannot be below than {min_value}')
                    continue

                if parameter_value > max_value:
                    print(f'{parameter} cannot be more than {min_value}')
                    continue


        setattr(config, parameter, parameter_value)


def init_train_config(proj_config) -> TrainerConfig:

    config = TrainerConfig()
    config.config.seed = proj_config.properties.seed
    config.data.seed = proj_config.properties.seed

    config.set_data(proj_config.data.prepared_data.training_data,
                    proj_config.data.prepared_data.test_data,
                    proj_config.data.dirs.model_saved_params_dir)


    change_train_config(config.data)
    change_train_config(config.config)


    return config



def change_train_params(configs: Configs):

    train_config = TrainerConfig(**configs.train_config)





def split_data(train_set, frac=0.1, seed=42):

    train_len = len(train_set)
    val_len = int(train_len * frac)

    rng = np.random.default_rng(seed=seed)
    val_idx = rng.choice(train_len, size=val_len, replace=False)

    return val_idx



