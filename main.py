import os
from cli.fonts import color_font
from cli import parse
import proj_func
from proj import func as proj_func_

import state
import app_state

import messages
from train import main as main_train

from proj import del_proj


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def mkproj(args):

    print(messages.INTRO)
    prj_name = args.name
    proj_func.make(prj_name, args)


def train_model(args):

    prj_name = args.name

    project = proj_func_.find_project(prj_name, state.APP_STATE)
    proj_config = app_state.load_state(app_state.CurrentProjectState, project.config_path)
    configs = proj_func_.load_proj_configs(proj_config)
    main_train.main(configs)



def delete_project(args):
    prj_name = args.name
    del_proj.delete_proj(prj_name)

    


def main(args):

    commands = {
        'mkproj': mkproj,
        'train': train_model,
        'delete': delete_project
    }

    if args.version:
        print(f'The version of GNN runner: {state.VERSION}')
        return None
    

    command = commands.get(args.command, None)

    if command:
        command(args)

    else:
        print(color_font('Invalid command!', 'red'))


if __name__ == '__main__':
    args = parse.make_parser()
    state.APP_STATE = app_state.init_global()
    main(args)


