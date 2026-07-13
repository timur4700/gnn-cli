import state
from cli import interactive
from proj import func
import shutil


def delete_proj(prj_name):

    if not state.APP_STATE.projects.projects:
        print('Where are no available projects to delete.')
        return None

    if not prj_name:

        available_projects = [k for k in state.APP_STATE.projects.projects.keys()]
        prj_name = interactive.query('Select project to delete: ', available_projects, False)

    func.find_project(prj_name, state.APP_STATE)

    directory = state.APP_STATE.projects.projects[prj_name]['wd']

    shutil.rmtree(directory)
    state.APP_STATE.projects.delete(prj_name)

    print(f'The project {prj_name} was successfully deleted.')

