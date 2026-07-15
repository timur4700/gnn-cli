from cli import interactive
from proj import func
from train.utils import change_train_params
from models.prep_model_config import change_model_params



def change_params(args, glob_state):

    prj_name = args.name

    

    if not prj_name:
        available_projects = [k for k in glob_state.projects.projects.keys()]
        prj_name = interactive.query('Select project, where you would like to change configuration: ', available_projects, False)

    configs = func.set_proj_configs(prj_name, glob_state)

    selector = ['Training Parameters', 'Model Parameters', 'Exit']

    commands = {'Training Parameters': change_train_params,
               'Model Parameters': change_model_params}

    while True:
        user_selection = interactive.query('What parameters would you like to change?', 
                                           selector,
                                           False)

        if user_selection == 'Exit':
            break

        commands[user_selection](configs)

    pass