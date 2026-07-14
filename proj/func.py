from pathlib import PosixPath
from utils import func
from typing import Any

import app_state

from graph_config import GraphPrepConfig
from train.config import TrainerConfig
from models.register import MODEL_REGISTER

from dataclasses import dataclass
from cli import interactive



@dataclass
class Configs:
    proj_config: app_state.CurrentProjectState
    graph_config: GraphPrepConfig
    model: Any
    model_config: Any
    train_config: TrainerConfig



def dir_check(project: app_state.ProjectGlobalInfo) -> bool:
    if not func.file_checker(project.wd):
            return False

    return True


def config_check(project: app_state.ProjectGlobalInfo) -> bool:
    if not func.file_checker(project.config_path):
        return False
    
    return True


def load_global_prj_info(prj_name: str,
                         appstate: app_state.AppState) -> app_state.ProjectGlobalInfo:

    project = appstate.projects.projects.get(prj_name, None)

    if not project:
        raise FileNotFoundError(f"The project {prj_name} is not registered.")

    if isinstance(project, dict):
        project = app_state.ProjectGlobalInfo(**project)

    return project


def find_project(prj_name, appstate: app_state.AppState) -> app_state.ProjectGlobalInfo:

    project = load_global_prj_info(prj_name, appstate)

    if not (dir_check(project) and config_check(project)):
        raise FileNotFoundError(f"The project {prj_name} does not contain global configuration file, or project directory")

    return project




def project_config(prj_path: PosixPath):

    config_path = prj_path / 'project_config.json'

    if func.file_checker(config_path):
        print('File exists.')

    else:
        pass


def configure_new_project(prj_name: str, appstate: app_state.AppState) -> app_state.CurrentProjectState:

    proj_config = app_state.CurrentProjectState(name=str(prj_name))

    proj_config.prepare_dirs_paths(appstate.files.dirs['saved_projects'])

    project_info = app_state.ProjectGlobalInfo(proj_config.working_dir,
                                               True,
                                               proj_config.config_path,
                                               func.calc_size(proj_config.working_dir))

    appstate.projects.update(prj_name, project_info)

    return proj_config


def inputs_checks(train_path: str, test_path: str, property_name: str, file_formats=['sdf']):

    if not train_path:
        train_path = func.file_user_input('ligand training molecules', file_formats)

    if not test_path:
        test_path = func.file_user_input('ligand test molecules', file_formats)

    if not property_name:
        property_name = func.looped_input('Enter the target property name for prediction: ', str)

    return train_path, test_path, property_name



def save_proj_configs(proj_config,
                      graph_config,
                      model_config,
                      train_config):


    graph_config.save(proj_config.graph_config_path)
    model_config.save(proj_config.model_params)
    train_config.save(proj_config.train_params)
    proj_config.save()


def proj_config_checks(proj_config: app_state.CurrentProjectState):

    path2check = (proj_config.graph_config_path,
                  proj_config.model_params,
                  proj_config.train_params)

    for file in path2check:
        if not func.file_checker(file):
            raise FileNotFoundError(f'The configuration file: {file}, was not found')




def load_proj_configs(proj_config: app_state.CurrentProjectState) -> Configs:

    proj_config_checks(proj_config)

    model_name = proj_config.model
    graph_config = app_state.load_state(GraphPrepConfig,
                                        proj_config.graph_config_path)

    model_config = app_state.load_state(MODEL_REGISTER[model_name].default_config,
                                        proj_config.model_params)

    model = MODEL_REGISTER[model_name].model

    train_config = app_state.load_state(TrainerConfig, 
                                        proj_config.train_params)

    return Configs(proj_config=proj_config,
                   graph_config=graph_config,
                   model=model,
                   model_config=model_config,
                   train_config=train_config)


def change_train_config(configs: Configs):
    pass



def change_config(proj_config: app_state.CurrentProjectState):
    configs2change = ['Model Config', 'Train Config']

    pass







    













