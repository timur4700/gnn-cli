import state
from utils import func

import random

from proj import func as func_proj, del_proj
import mol_utils
import graph_prep
import graph_config
from models import prep_model_config

from train.utils import init_train_config



def make(prj_name: str, args):


            if not prj_name:
                prj_name = func.looped_input('Enter the project name\n', str)

            prj_name = prj_name.strip()

            if str(prj_name) in state.APP_STATE.projects.projects:
                raise FileExistsError((f'The project with the name ({prj_name}) already exists.\nPlease choose another name.'))
                   
            
                
            proj_config = func_proj.configure_new_project(prj_name, state.APP_STATE)


            try:
                # Loading train and test mols paths
                train_path, test_path, prop_name = func_proj.inputs_checks(args.train_path,
                                                                            args.test_path,
                                                                            args.property)

                protein_path = ''

                proj_config.properties.seed = random.randint(0, 1000000)

### TODO: Add optional protein input
###            if interactive.query('Include protein?', ['No', 'Yes']):
###                protein_path = func.file_user_input('protein', ['sdf'])

                proj_config.set_mols_n_prop(train_path,
                                            test_path,
                                            prop_name,
                                            protein_path)

                


                data = mol_utils.main(proj_config)

                graph_prep_config = graph_config.configure_graph_preparation(proj_config)

                train_data, test_data, graph_prep_config = graph_prep.main(data, 
                                                        proj_config, 
                                                        graph_prep_config)


                model_config = prep_model_config.init_model_configuration(proj_config, 
                                                                        graph_prep_config)
                
                train_config = init_train_config(proj_config)

                func_proj.save_proj_configs(proj_config, 
                                            graph_prep_config, 
                                            model_config, 
                                            train_config)

                print(f'The project {str(prj_name)} has been successfully prepared!')

            except Exception as e:
                  print(f'During preparation an error occured: {e}')
                  del_proj.delete_proj(prj_name)

            except KeyboardInterrupt:
                  print()
                  del_proj.delete_proj(prj_name)
                  