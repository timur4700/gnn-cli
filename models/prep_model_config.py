from models.config_defaults import ConfigStorage, ModelConfig
from graph.graph_config import GraphPrepConfig
from dataclasses import asdict, Field, fields
from cli import interactive
from utils import func
from rich.table import Table
from rich.console import Console
from models.register import MODEL_REGISTER
from utils.func import load_json
from app_state import load_state



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
                             graph_config: GraphPrepConfig) -> ConfigStorage:

    model = model_choice(graph_config.graph_type)
    config = MODEL_REGISTER[model].default_config(input_dim=graph_config.graph_dims.input_dim,
                                                  edge_dim=graph_config.graph_dims.edge_dim)

    proj_config.model = model
    change_model_config(config)

    proj_model_config = ConfigStorage({
        'model_config_id_1': ModelConfig(config=config)
    })

    return proj_model_config


def load_model_config(proj_config):

    model_name = proj_config.model
    config_path = proj_config.model_params
    model_configs = ConfigStorage(cur_proj_configs=load_json(config_path)['cur_proj_configs'])

    model = MODEL_REGISTER[model_name].model
    default_params = MODEL_REGISTER[model_name].default_config

    available_params = [k for k in model_configs.cur_proj_configs.keys()]
    config_name = interactive.query('Choose model parameters: ', available_params, False)

    model_params = default_params(**model_configs.cur_proj_configs[config_name]['config'])


    return config_name, model, model_params




def change_model_params(proj_config):

    model_config = load_state(ConfigStorage, proj_config.model_params)
    model_name = proj_config.model

    num_configs = len(model_config.cur_proj_configs)

    if  num_configs >= 1:
        print(f'For the project [{proj_config.name}] found one or more configs')

        choices = ['Change', 'Create New Config']
        create = interactive.query('Whould you like to change one of the current configs, or create new one based on default?',
                                 choices)

        config_name = f'model_config_id_' + str(num_configs + 1)

        if create:
            graph_dims = load_json(proj_config.graph_config_path)["graph_dims"]
            config = MODEL_REGISTER[model_name].default_config(input_dim=graph_dims['input_dim'],
                                                                       edge_dim=graph_dims['edge_dim'])

        else:

            _, _, config = load_model_config(proj_config)

            
        change_model_config(config)

        model_config.cur_proj_configs[config_name] = ModelConfig(config)
        model_config.save(proj_config.model_params)
        print(f"Model {'new' if create else 'changed'} config saved as [{config_name}]")




