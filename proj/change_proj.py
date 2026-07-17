from cli import interactive
from proj import func
from train.utils import change_train_params
from models.prep_model_config import change_model_params
import app_state


def change_params(args, glob_state):

    prj_name = args.name

    if not prj_name:
        available_projects = [k for k in glob_state.projects.projects.keys()]
        prj_name = interactive.query('Select project, where you would like to change configuration: ', available_projects, False)

    project = func.find_project(prj_name, glob_state)
    proj_config = app_state.load_state(app_state.CurrentProjectState, project.config_path)

    selector = ['Training Parameters', 'Model Parameters', 'Exit']

    commands = {'Training Parameters': change_train_params,
               'Model Parameters': change_model_params}

    while True:
        user_selection = interactive.query('Which parameters would you like to change?', 
                                           selector,
                                           False)

        if user_selection == 'Exit':
            break

        commands[user_selection](proj_config)

    pass


def change_glob_config(args, glob_state):

    prj_name = args.name