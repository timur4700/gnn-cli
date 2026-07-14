from rdkit import Chem
from typing import Union, Tuple
from functools import partial
from dataclasses import asdict
import torch
from torch_geometric.data import Data, HeteroData
import mol_encoding

import numpy as np
from multiprocessing import Pool
import time
import app_state


def radius(x, y, r):

    diff = x[:, None, :] - y[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    src, dst = np.where(distances <= r)

    return np.vstack((src, dst))


def add_index(edge_list: bool, *args: torch.Tensor):

    dim = -1 if edge_list else 0

    indexed_tensor = []

    for i, tensor in enumerate(args):
        device = tensor.device
        dtype = tensor.dtype
        idx = torch.tensor([i]*tensor.shape[dim], device=device, dtype=dtype)

        if edge_list:
            indexed_tensor.append(idx.reshape(-1, 1))
            continue

        indexed_tensor.append(torch.hstack((tensor, idx.reshape(-1, 1))))

    return indexed_tensor


def prep_union_graph(data: Data,
                     x_lig,
                     x_prot,
                     edge_lig,
                     edge_prot,
                     inter_cutoff):

    x_lig, x_lig_pos = x_lig
    x_prot, x_prot_pos = x_prot

    x_lig = torch.from_numpy(x_lig).float()
    x_prot = torch.from_numpy(x_prot).float()

    edge_lig = torch.from_numpy(edge_lig).long()
    edge_prot = torch.from_numpy(edge_prot).long()

    edge_inter = torch.from_numpy(
        radius(x_lig_pos, x_prot_pos, inter_cutoff)
    ).long()

    edge_attr = torch.cat(add_index(True, edge_lig, edge_prot, edge_inter), dim=0)
    edge_index = torch.cat([edge_lig, edge_prot, edge_inter], dim=-1)

    x = torch.cat(add_index(False, x_lig, x_prot), dim=0)

    x_lig_pos = torch.from_numpy(x_lig_pos).float()
    x_prot_pos = torch.from_numpy(x_prot_pos).float()

    data.x = x
    data.pos = torch.cat([x_lig_pos, x_prot_pos], dim=0)
    data.edge_index = edge_index
    data.edge_attr = edge_attr

    return data



def prep_hetero_graph(x_lig, 
                      x_prot,
                      edge_lig,
                      edge_prot,
                      encode_coords,
                      bond_features) -> HeteroData:

    data = HeteroData()

    if encode_coords:
        x_lig, x_lig_pos = x_lig
        x_prot, x_prot_pos = x_prot

    if bond_features:
        edge_lig, edge_lig_attr = edge_lig
        edge_prot, edge_prot_attr = edge_prot

    data['ligand'].x = torch.from_numpy(x_lig).float()
    data['protein'].x = torch.from_numpy(x_prot).float()

    data['ligand','to','ligand'].edge_index = torch.from_numpy(edge_lig).long()
    data['protein','to','protein'].edge_index = torch.from_numpy(edge_prot).long()

    if encode_coords:
        data['ligand'].pos = torch.from_numpy(x_lig_pos).float()
        data['protein'].pos = torch.from_numpy(x_prot_pos).float()

    if bond_features:
        data['ligand','to','ligand'].edge_attr = torch.from_numpy(edge_lig_attr).float()
        data['protein','to','protein'].edge_attr = torch.from_numpy(edge_prot_attr).float()

    return data



def double_featurization(func, *args: Chem.Mol):

    features = []

    for mol in args:
        features.append(func(mol))

    return features



def single_graph_prep(mol: Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                     target_y,
                     onehot_encoding: bool=False,
                     atom_dict: dict=None,
                     atom_features: list[str]=[],
                     encode_coords: bool=False,
                     chemical_based_topology: bool=True,
                     bond_features: bool=False,
                     dist: bool=False,
                     include_interactions=False,
                     intra_cutoff=5,
                     inter_cutoff=5) -> Union[Data, HeteroData]:
    data = Data()

    lig_prot_data = isinstance(mol, tuple)

    if lig_prot_data:
        name = mol[0].GetProp('_Name').split('_')[0]

    else:
        name = mol.GetProp('_Name').split('_')[0]


    y = torch.tensor(target_y, dtype=torch.float32, device='cpu')
    data['y'] = target_y
    data.pdb_id = name

    if include_interactions and lig_prot_data:
        encode_coords = True
        bond_features = False

    
    node_feature_func = partial(mol_encoding.MolFeaturization.atom_featurization, 
                                features=atom_features, 
                                onehot=onehot_encoding, 
                                get_pos=encode_coords,
                                atom_dict=atom_dict)



    edge_index_func = (partial(mol_encoding.MolFeaturization.get_edge_list_chemical_topology, 
                              dist=dist,
                              add_features=bond_features,
                              onehot=onehot_encoding) if chemical_based_topology else 
                              partial(mol_encoding.MolFeaturization.get_edge_list_spatial_topology, cutoff=intra_cutoff))



    if lig_prot_data:
        x_lig, x_prot = double_featurization(node_feature_func, mol[0], mol[1])
        edge_lig, edge_prot = double_featurization(edge_index_func, mol[0], mol[1])



        if not include_interactions:
            data = prep_hetero_graph(x_lig, 
                                    x_prot, 
                                    edge_lig, 
                                    edge_prot, 
                                    encode_coords=encode_coords,
                                    bond_features=bond_features)

            #data['pdb_id'] = name
            data.y = y
            return data

        data = prep_union_graph(data, x_lig, x_prot, edge_lig, edge_prot, inter_cutoff)
        #data.pdb_id = name
        return data

    x = node_feature_func(mol)
    edge_data = edge_index_func(mol)



    if encode_coords:
            data.x, data.pos = (torch.tensor(x[0], dtype=torch.float32, device='cpu'), 
                                torch.tensor(x[1], dtype=torch.float32, device='cpu'))

    else:
            data.x = torch.tensor(x, dtype=torch.float32, device='cpu')

    if bond_features:
            data.edge_index, data.edge_attr = (torch.tensor(edge_data[0], dtype=torch.long, device='cpu'), 
                                            torch.tensor(edge_data[1], dtype=torch.float32, device='cpu'))

    else:
            data.edge_index = torch.tensor(edge_data, dtype=torch.long, device='cpu')

    return data


def check_dimensions(data: Data | HeteroData):

    if isinstance(data, Data):
        node_dim = data.x.size(-1)

        if data.edge_attr is not None:
            edge_dim = data.edge_attr.size(-1)
            return node_dim, edge_dim
        
        return node_dim, 0

    if isinstance(data, HeteroData):
        node_dim = data['ligand'].x.size(-1)

        if data['ligand', 'to', 'ligand'].edge_attr is not None:
            edge_dim = data['ligand', 'to', 'ligand'].edge_attr.size(-1)
            return node_dim, edge_dim
        
        return node_dim, 0
         


def main(mol_data, config: app_state.CurrentProjectState, graph_config):


    train_data_path, test_data_path = (config.data.prepared_data.training_data,
                                       config.data.prepared_data.test_data)

    atom_dict = mol_data['atom_codes']
    graph_config_dict = asdict(graph_config.graph_config)
    graph_config_dict['atom_dict'] = atom_dict

    func = partial(single_graph_prep, 
                   **graph_config_dict)

    start = time.perf_counter()
    train_data: list[Data]=[func(mol, y, atom_dict=atom_dict) for mol, y in 
                            zip(mol_data['train']['mols'], mol_data['train']['target'])]
    
    test_data: list[Data]=[func(mol, y, atom_dict=atom_dict) for mol, y in 
                            zip(mol_data['test']['mols'], mol_data['test']['target'])]

    node_dim, edge_dim = check_dimensions(train_data[0])

    graph_config.graph_dims.input_dim = node_dim
    graph_config.graph_dims.edge_dim = edge_dim

    finish = time.perf_counter()
    print(f'Graph Preparation Execution time: {finish - start}')

    torch.save(train_data, train_data_path)
    torch.save(test_data, test_data_path)
    print(f'The training and test data successfuly saved in {train_data_path} and {test_data_path}, respectively')


    return train_data, test_data, graph_config