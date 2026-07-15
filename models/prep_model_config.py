from models import config_defaults
from graph.graph_config import GraphPrepConfig
from dataclasses import asdict, Field, fields
from cli import interactive
from utils import func
from rich.table import Table
from rich.console import Console
from models.register import MODEL_REGISTER

from proj.func import Configs



def model_search(graph_type):

    available_models = [model for model, v in MODEL_REGISTER.items()
                    if graph_type in v.graph_types]

    return available_models


def model_choice(graph_type):
    aval_models = model_search(graph_type)

    choice = interactive.query('Select the model, which fits to your graph:', aval_models, False)
    return choice




def print_model_config(config,
                       title: str='Model Configuration'):
    console = Console()
    model_table = Table(title=title)
    model_table.add_column('Parameter')
    model_table.add_column('Default Value')
    model_table.add_column('Current Value')

    for field in fields(config):
        name = field.name
        default = field.default
        current = getattr(config, name)

        if name == 'register_name':
            continue


        model_table.add_row(name, str(default), str(current))    
    console.print(model_table)


def change_model_config(config):

    choices = [field.name for field in fields(config) if field.name != 'register_name'] + ['Confirm']
 
    while True:

        print_model_config(config)
        parameter = interactive.query('Choose model parameter', choices, False)

        if parameter == 'Confirm':
            break


        field: Field = config.__dataclass_fields__[parameter]

        if not field.metadata['editable']:
            print(f'You cannot change the parameter {parameter}')
            continue

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




def init_model_configuration(proj_config,
                             graph_config: GraphPrepConfig):

    model = model_choice(graph_config.graph_type)
    config = MODEL_REGISTER[model].default_config(input_dim=graph_config.graph_dims.input_dim,
                                                  edge_dim=graph_config.graph_dims.edge_dim)

    proj_config.model = model
    change_model_config(config)

    return config


def change_model_params(configs: Configs):

    model_config = configs.model_config

    change_model_config(model_config)
    model_config.save(configs.proj_config.model_params)



