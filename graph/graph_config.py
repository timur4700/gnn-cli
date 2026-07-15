from dataclasses import dataclass, asdict, field
from cli import interactive
from utils import func
from typing import List
from functools import partial
import messages




ATOM_FEATURES = {
                'Atomic Number': 'atomic_num',
                'Atom Charge': 'atom_charge',
                'Aromaticity': 'aromaticity',
                'Number of Hydrogens': 'num_hydrogen',
                'Atom Hybridization': 'atom_hybrid',
                'Atomic Mass': 'atom_mass',
                'Atom Valence': 'atom_valence',
                'Atom Degree': 'atom_degree'
                }


@dataclass
class GraphDims:
    input_dim: int=0
    edge_dim: int=0


@dataclass
class Config:
    atom_features: List[str]=field(default_factory=lambda: [])
    onehot_encoding: bool=False
    encode_coords: bool=False
    chemical_based_topology: bool=False
    bond_features: bool=False
    dist: bool=False
    include_interactions: bool=False
    intra_cutoff: int | float=5
    inter_cutoff: int | float=5






@dataclass
class GraphPrepConfig:
    graph_config: Config=field(default_factory=Config)
    graph_dims: GraphDims=field(default_factory=GraphDims)
    graph_type: str='ligand_only'

    def save(self, wd):
        func.save_json(asdict(self), wd)


class Choices:
    graph_represent: List[str] = ['Chemical Topology (2D)', 'Spatial Topology (3D)']
    features_type: List[str] = ['Standard Features', 'One-hot feature encoding']
    binary: List[str] = ['No', 'Yes']





def studied_feat_set(): 
    return ['Atomic Number', 'Aromaticity', 'Number of Hydrogens']


def all_feat_set(avail_set: list):
    return avail_set


def print_feature_list(avail_set, included_set):
    output = '\n'.join([f"[x] {feat}" if feat in included_set else f"[ ] {feat}" for feat in avail_set])
    print('INCLUDED FEATURES:')
    print('------------------')
    print(output + '\n')




def custom_feat_set(avail_set: list, included_set: list):
    avail_set_orig = avail_set.copy()
    user_selections = ['Confirm Selections', 'Reset Selections']

    print_feature_list(avail_set, included_set)

    while True:
        
        user_input = interactive.query('Choose atomic feature to include: ', avail_set + user_selections, False)

        if user_input == user_selections[0]:
            if not included_set:
                print('Features were not selected. All atomic features will be included.')
                return avail_set

            return included_set

        elif user_input == user_selections[1]:
            included_set.clear()
            avail_set = avail_set_orig.copy()

        else:            
            included_set.append(avail_set.pop(avail_set.index(user_input)))

        print_feature_list(avail_set_orig, included_set)
 


def include_atomic_features():
    global ATOM_FEATURES

    included_features = []
    available_features = [k for k in ATOM_FEATURES]


    global_choice = {
        'Studied Features Set': studied_feat_set,
        'All Features Set': partial(all_feat_set, available_features),
        'Custom Features Set': partial(custom_feat_set, avail_set=available_features, included_set=included_features)

    }

    user_choice = interactive.query('Choose feature set', list(global_choice.keys()), False)

    included_features = global_choice[user_choice]()
    included_features = [ATOM_FEATURES[i] for i in included_features]

    return included_features



def configure_graph_preparation(proj_config) -> GraphPrepConfig:

    config = GraphPrepConfig()
    choice_variants = Choices()

    print(messages.GRAPH_PREP_START_MSG)
    print(messages.AVAIL_ATOM_FEATURES)

    config.graph_config.atom_features = include_atomic_features()
    config.graph_config.onehot_encoding = interactive.query('Choose feature encoding method:', choice_variants.features_type) == 1
    config.graph_config.chemical_based_topology = interactive.query('Choose Graph representation type:', choice_variants.graph_represent) == 0


    if config.graph_config.chemical_based_topology:
        config.graph_config.encode_coords = interactive.query('Encode atomic postions?', choice_variants.binary) == 1
        config.graph_config.bond_features = interactive.query('Include bond features?', choice_variants.binary) == 1

        if config.graph_config.bond_features:
            config.graph_config.dist = interactive.query('Include intra-atomic distances (A)?', choice_variants.binary) == 1

    else:
        print('For Spatial graph representation bond features are turned off.')
        config.graph_config.encode_coords = True
        config.graph_config.intra_cutoff = func.looped_input('Enter intra-molecular atomic distance cutoff (A): ', float)

    if proj_config.data.mols_data.protein_mols:

        config.graph_type = 'plc_no_inter'

        if interactive.query('Include interaction edges between ligand and protein?', choice_variants.binary) == 1:
            config.graph_config.inter_cutoff = func.looped_input('Enter inter-molecular atomic distance cutoff (A): ', float)
            config.graph_type = 'plc_inter'

    return config
