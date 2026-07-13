from models import config_defaults, ligand_model
from dataclasses import dataclass
from typing import Type, List




@dataclass(frozen=True)
class Model:
    model: Type
    default_config: Type
    graph_types: List[str]



MODEL_REGISTER = {
    'ligand_base_model': Model(model=ligand_model.LigandModel,
                               default_config=config_defaults.LigandBaseModelConfig,
                               graph_types=['ligand_only'])
                }