from rdkit import Chem
import numpy as np
from typing import List, Dict, Tuple, Union
from app_state import CurrentProjectState



general_ligands = 'mol_data/misato_train_ligands_h.sdf'
core_ligands = 'mol_data/misato_core_ligands_h.sdf'
proteins_all = 'mol_data\misato_all_pockets.sdf'


def load_sdf(path: str, mol_type:str='', removehs=True) -> List[Chem.Mol]:
    supplier = Chem.SDMolSupplier(path, removeHs=removehs)
    init_num = len(supplier)

    mols = [mol for mol in Chem.SDMolSupplier(path) if mol]
    final_num = init_num - len(mols)

    print(f'The number of {mol_type} molecules excluded due to errors: {final_num}')

    return mols


def combine_protein(train_mols: list, test_mols: list, protein_mols_all: list):

    train_mols_lookup = {mol.GetProp('_Name').split('_')[0]:mol for mol in train_mols}
    test_mols_lookup = {mol.GetProp('_Name').split('_')[0]:mol for mol in test_mols}

    train_mols_comb = []
    test_mols_comb = []

    for protein_mol in protein_mols_all:

        prot_name = protein_mol.GetProp('_Name').split('_')[0].lower()

        mol = train_mols_lookup.get(prot_name, None)
        if mol is not None:
            train_mols_comb.append((mol, protein_mol))
            continue

        mol = test_mols_lookup.get(prot_name, None)
        if mol is not None:
            test_mols_comb.append((mol, protein_mol))


    print(f"Number of prepared training Protein-Ligand complexes: {len(train_mols_comb)}\n"
          f"Number of prepared test Protein-Ligand complexes: {len(test_mols_comb)}")


    return train_mols_comb, test_mols_comb


def get_target_prop(mols: List[Chem.Mol], prop_name):

    def get_mol_prop(mol, prop_name):
        if isinstance(mol, tuple):
            mol = mol[0]

        return mol.GetDoubleProp(prop_name)

    targets = [get_mol_prop(mol, prop_name) for mol in mols]

    return targets


def prep_sdf(train_sdf_path: str, test_sdf_path, 
              prop_name: str, protein_path=None) -> Dict[str, Dict[str, Union[List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]],np.ndarray]]]:

    protein = True if protein_path else False
     
    train_mols = load_sdf(train_sdf_path, 'training')
    print(f'Number of loaded training molecules: {len(train_mols)}')

    test_mols = load_sdf(test_sdf_path, 'test')
    print(f'Number of loaded test molecules: {len(test_mols)}')


    mol_data = {'train': {
                         'mols': train_mols,
                         },

                'test': {
                         'mols': test_mols,             
                        }
               }

    if protein:
        assert protein_path
        protein_mols = load_sdf(protein_path, 'protein')
        print(f'Number of loaded protein molecules: {len(protein_mols)}\n')

        mol_data['train']['mols'], mol_data['test']['mols'] = combine_protein(train_mols, 
                                                test_mols, 
                                                protein_mols)

    mol_data['train']['target'] = get_target_prop(train_mols, prop_name)
    mol_data['test']['target'] = get_target_prop(train_mols, prop_name)


    return mol_data



def main(proj_state: CurrentProjectState) -> Dict[str, Dict[str, Union[List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]],np.ndarray]]]:
    train_path, test_path, protein_path, prop_name = proj_state.get_mols_n_prop()

    data = prep_sdf(train_path, 
                    test_path, 
                    prop_name,
                    protein_path)


    return data


