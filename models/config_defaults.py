from dataclasses import dataclass, field, asdict
from models import defaults_params
from typing import List, Dict, Type, Union

from utils.func import save_json


@dataclass
class ConfigStorage:
    cur_proj_configs: Dict[str, Type]



@dataclass
class ModelConfig:
    config: Type | Dict[str, Union[int, float, str, bool]]
    saved_params: List[str]
    


@dataclass
class LigandBaseModelConfig:

    input_dim: int=field(default=1,
                             metadata={
                                        'editable': False
                                    })

    hidden_dim: int=field(default=64,
                         metadata={
                             'min': 1,
                             'max': 2**16,
                             'editable': True
                             
                         })
    
    output_dim: int=field(default=1,
                         metadata={
                             'min': 1,
                             'max': 2**16,
                             'editable': True

                         })

    mpnn_type: str=field(default='gcn',
                         metadata={
                             'choices': defaults_params.MPNN,
                             'editable': True
                         })
    
    edge_attr: bool=field(default=False,
                         metadata={
                             'choices': [False, True],
                             'editable': True
                         })
    edge_dim: int=field(default=0,
                         metadata={
                             'min': 0,
                             'max': 2**16,
                             'editable': False
                         })
    num_mpnn_layers: int=field(default=1,
                         metadata={
                             'min': 1,
                             'max': 2**16,
                             'editable': True
                         })

    act_function: str=field(default='relu',
                            metadata={
                                'choices': defaults_params.ACTIVATION_FUNCTIONS,
                                'editable': True
                            })
    
    glob_pool: str=field(default='sum',
                         metadata={
                             'choices': defaults_params.GLOBAL_POOLING,
                             'editable': True
                         })

    def save(self, wd):
        save_json(asdict(self), wd)

        
        