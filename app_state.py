from dataclasses import dataclass, field, asdict
from typing import ClassVar, Dict, Type, List
from utils.func import load_json, save_json, file_checker
from cli.interactive import query
from pathlib import Path, PosixPath
import os
import json

from torch.cuda import is_available

@dataclass
class GlobalProjectState:

    last_project: str=None
    device: str='cpu'
    n_proc: int=1
    data_sizes: int | float=0

    def save(self, path: str | PosixPath):
        data = vars(self)
        save_json(data, path)



@dataclass
class CurrentProjectProperties:
    seed: int=42
    property_name: str=''


@dataclass
class CurrentProjectDirs:
    model_prepared_data_dir: str='prepared_data'
    sdf_data: str='sdf_files'
    model_saved_params_dir: str='model_parameters'

@dataclass
class CurrentProjectFiles:
    model_params_path: str='model_params.json'
    train_params_path: str='train_params.json'


@dataclass
class CurrentProjectMols:
    training_mols: str=''
    test_mols: str=''
    protein_mols: str=''



@dataclass
class CurrentProjectGraphs:
    training_data: str='train.pt'
    test_data: str='test.pt'


@dataclass
class CurrentProjectData:
    dirs: CurrentProjectDirs=field(default_factory=CurrentProjectDirs)
    state_data: CurrentProjectFiles=field(default_factory=CurrentProjectFiles)
    mols_data: CurrentProjectMols=field(default_factory=CurrentProjectMols)
    prepared_data: CurrentProjectGraphs=field(default_factory=CurrentProjectGraphs)

 


@dataclass
class CurrentProjectState:
    name: str=None
    working_dir: str=None
    config_path: str=None
    properties: CurrentProjectProperties=field(default_factory=CurrentProjectProperties)
    data: CurrentProjectData=field(default_factory=CurrentProjectData)
    graph_config_path: str=None
    model: str=None
    model_params: dict=None
    train_params: dict=None

    def prepare_dirs_paths(self, storage):
        assert self.name, 'Project name was not given.'
        self.working_dir = str(Path(storage) / self.name)
        self.config_path = str(Path(self.working_dir)/'settings.json')
        self.graph_config_path = str(Path(self.working_dir)/'graph_config.json')
        self.model_params = str(Path(self.working_dir)/'model_config.json')
        self.train_params = str(Path(self.working_dir)/'train_config.json')

        if not file_checker(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)

        self.set_path(self.data.dirs, self.working_dir, make_dir=True)
        self.set_path(self.data.state_data, self.working_dir)
        self.set_path(self.data.prepared_data, self.data.dirs.model_prepared_data_dir)



    def set_path(self, object ,storage, make_dir=False):
        for k, v in vars(object).items():
            path = Path(storage)/v
            setattr(object, k, str(path))

            if make_dir:
                if not file_checker(path):
                    os.makedirs(path, exist_ok=True)
                
            
    def convert_json(self):
        data = json.dumps(asdict(self), indent=4)
        return data
    

    def save(self):
        assert self.config_path

        save_json(asdict(self), self.config_path)


    def set_mols_n_prop(self, 
                 train_mols: str, 
                 test_mols: str, 
                 prop_name: str,
                 protein_mols: str=''):

        self.data.mols_data.training_mols = train_mols
        self.data.mols_data.test_mols = test_mols
        self.properties.property_name = prop_name
        self.data.mols_data.protein_mols = protein_mols


    def get_mols_n_prop(self) -> List[str]:

        paths = [v for k, v in vars(self.data.mols_data).items()] + [self.properties.property_name]
        return paths



@dataclass
class FileNamesGlobal:
    dirs: ClassVar[dict[str,str]] = {'saved_data': 'app_data',
                  'saved_projects': 'app_data/saved_projects'}
    global_config: ClassVar[str]='app_data/global_config.json'

    def prepare_dirs(self):
        for k, v in self.dirs.items():
            if not file_checker(v):
                os.makedirs(v, exist_ok=True)

        


@dataclass
class ProjectGlobalInfo:
    wd: str=''
    exists: bool=False
    config_path: str=''
    size: int | float=0


@dataclass
class CurrentProjects:
    file_name: str='app_data/current_projects.json'
    projects: Dict[str, ProjectGlobalInfo]=field(default_factory=dict)


    def save(self):
        save_json(asdict(self), self.file_name)


    def start(self):
        if not file_checker(self.file_name):
            self.save()


    def update(self, prj_name: str, state: ProjectGlobalInfo):
        self.projects[prj_name] = state
        self.save()


    def delete(self, prj_name):
        del self.projects[prj_name]
        self.save()



@dataclass
class AppState:
    files: FileNamesGlobal = field(default_factory=FileNamesGlobal)
    settings: GlobalProjectState = field(default_factory=GlobalProjectState)
    projects: CurrentProjects = field(default_factory=CurrentProjects)




def load_state(state_cls: Type[GlobalProjectState | 
                               CurrentProjectState |
                               CurrentProjects |
                               AppState], 
               path: str) -> CurrentProjectState | CurrentProjects | AppState:

    if file_checker(path):

        state = load_json(path)
        return state_cls(**state)

    else:
        data = asdict(state_cls())
        save_json(data, path)

        return state_cls()



def check_registered_projects(state: AppState):

    to_delete = []

    for prj_name, v in state.projects.projects.items():
        if not file_checker(v['wd']):
            print(f'The main folder for {prj_name} was not found.')
            user_answer = query('Do you want delete it from global register', ['No', 'Yes'])

            if user_answer:
                to_delete.append(prj_name)

    for prj_del in to_delete:
        state.projects.delete(prj_del)
            


def init_global():

    device = 'cuda' if is_available() else 'cpu'

    glob_state = GlobalProjectState(device=device)

    app = AppState(settings=glob_state)
    app.files.prepare_dirs()

    path = app.files.global_config

    app.settings = load_state(GlobalProjectState, path)
    app.projects = load_state(CurrentProjects, app.projects.file_name)

    check_registered_projects(app)

    return app


