from rdkit import Chem
import numpy as np
from typing import List
from functools import partial


class StandardEncoding:
        @staticmethod
        def atomic_num(atom: Chem.rdchem.Atom, atom_dict=None):

            if atom_dict:
                return atom_dict[atom.GetSymbol()]

            return atom.GetAtomicNum()
        
        @staticmethod
        def aromaticity(atom: Chem.rdchem.Atom):
                aromatic = 1 if atom.GetIsAromatic() else 0
                return aromatic
        
        @staticmethod
        def hydrogen_count(atom: Chem.rdchem.Atom):
                return atom.GetTotalNumHs()
        
        @staticmethod
        def hybridization(atom: Chem.rdchem.Atom):

                possible_hybridization_list = [
                                                Chem.rdchem.HybridizationType.S,
                                                Chem.rdchem.HybridizationType.SP,
                                                Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3,
                                                Chem.rdchem.HybridizationType.SP3D,
                                                Chem.rdchem.HybridizationType.SP3D2,
                                                Chem.rdchem.HybridizationType.UNSPECIFIED
                                        ]
                try:
                        hyb_type = possible_hybridization_list.index(atom.GetHybridization())

                except:
                        hyb_type = 6

                return hyb_type
        

        @staticmethod
        def atom_mass(atom: Chem.rdchem.Atom):
                return atom.GetMass()
        
        @staticmethod
        def atom_charge(atom: Chem.rdchem.Atom):
                possible_charges = [-2, -1, 0, 1, 2, 3]
                return possible_charges.index(atom.GetFormalCharge())
        
        @staticmethod
        def atom_valence(atom: Chem.rdchem.Atom):
                return atom.GetTotalValence()
        
        @staticmethod
        def atom_degree(atom: Chem.rdchem.Atom):
                return atom.GetDegree()
        

        @staticmethod
        def aromaticity_bond(bond: Chem.rdchem.Bond):
                arom = 1 if bond.GetIsAromatic() else 0

                return arom

        @staticmethod
        def bond_type(bond: Chem.rdchem.Bond):

                possible_bonds_type = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                                        Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.ZERO,
                                        Chem.rdchem.BondType.OTHER]
    
    
                try:
                        bond_t = possible_bonds_type.index(bond.GetBondType())
                except:
                        bond_t = 6

                return bond_t
    
        @staticmethod
        def bond_stereo(bond: Chem.rdchem.Bond):
                possible_stereo = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOATROPCW, Chem.rdchem.BondStereo.STEREOATROPCCW,
                                        Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS, 
                                        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
                

                stereo = possible_stereo.index(bond.GetStereo())

                return stereo

        
class OneHotAtomFeatures:
        @staticmethod
        def atomic_num(atom: Chem.rdchem.Atom, atom_dict: dict):
                size = len(atom_dict)
                vector = np.zeros(shape=(size,))
                idx = atom_dict[atom.GetSymbol()]
                vector[idx] = 1
                return vector

        @staticmethod
        def aromaticity(atom: Chem.rdchem.Atom):
                size = 2
                vector = np.zeros(shape=(size,))
                vector[int(StandardEncoding.aromaticity(atom))] = 1

                return vector
        
        @staticmethod
        def hydrogen_count(atom: Chem.rdchem.Atom):
                size = 5
                vector = np.zeros(shape=(size,))
                vector[int(StandardEncoding.hydrogen_count(atom))] = 1

                return vector
        
        @staticmethod
        def hybridization(atom: Chem.rdchem.Atom):
                size = 7
                vector = np.zeros(shape=(size,))
                vector[int(StandardEncoding.hybridization(atom))] = 1

                return vector
        
        @staticmethod
        def atom_valence(atom: Chem.rdchem.Atom):
                size = 6

                vector = np.zeros(shape=(size,))
                vector[int(StandardEncoding.atom_valence(atom)-1)] = 1

                return vector

        @staticmethod
        def atom_degree(atom: Chem.rdchem.Atom):
                size = 6
                vector = np.zeros(shape=(size,))
                vector[int(StandardEncoding.atom_degree(atom)-1)] = 1

                return vector
        
        @staticmethod
        def atom_charge(atom: Chem.rdchem.Atom):
                size = 11
                vector = np.zeros(shape=(size,))
                vector[int(StandardEncoding.atom_charge(atom)+5)] = 1
                return vector
        

        @staticmethod
        def aromaticity_bond(bond: Chem.rdchem.Bond):
                size = 2
                vector = np.zeros(shape=(size,))
                arom = 1 if bond.GetIsAromatic() else 0
                vector[arom] = 1

                return vector

        @staticmethod
        def bond_type(bond: Chem.rdchem.Bond):

                possible_bonds_type = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                                        Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.ZERO,
                                        Chem.rdchem.BondType.OTHER]
    
                size = len(possible_bonds_type)
                vector = np.zeros(shape=(size,))
    
                try:
                        bond_t = possible_bonds_type.index(bond.GetBondType())
                except:
                        bond_t = 6

                vector[bond_t] = 1
                return vector
    
        @staticmethod
        def bond_stereo(bond: Chem.rdchem.Bond):
                possible_stereo = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOATROPCW, Chem.rdchem.BondStereo.STEREOATROPCCW,
                                        Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS, 
                                        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
                
                size = len(possible_stereo)
                vector = np.zeros(shape=(size,))

                stereo = possible_stereo.index(bond.GetStereo())
                vector[stereo] = 1

                return vector




class MolFeaturization:
        @staticmethod
        def atom_featurization(mol: Chem.Mol, features: List[str]=None, onehot=False, atom_dict=None, get_pos=False):

                if get_pos:
                    conformer = mol.GetConformer()
                    pos = conformer.GetPositions()
                       
                       

                node_featurers = {
                        'atomic_num': StandardEncoding.atomic_num,
                        'atom_charge': StandardEncoding.atom_charge,
                        'aromaticity': StandardEncoding.aromaticity,
                        'num_hydrogen': StandardEncoding.hydrogen_count,
                        'atom_hybrid': StandardEncoding.hybridization,
                        'atom_mass': StandardEncoding.atom_mass,
                        'atom_charge': StandardEncoding.atom_charge,
                        'atom_valence': StandardEncoding.atom_valence,
                        'atom_degree': StandardEncoding.atom_degree
                }

                if onehot:
        
                        assert atom_dict, 'Please, provide atom dictionary for proper one-hot atomic featurization.'

                        node_featurers.update(
                                {
                                'atomic_num': partial(OneHotAtomFeatures.atomic_num, atom_dict=atom_dict),
                                'atom_charge': OneHotAtomFeatures.atom_charge,
                                'aromaticity': OneHotAtomFeatures.aromaticity,
                                'num_hydrogen': OneHotAtomFeatures.hydrogen_count,
                                'atom_hybrid': OneHotAtomFeatures.hybridization,
                                'atom_charge': OneHotAtomFeatures.atom_charge,
                                'atom_valence': OneHotAtomFeatures.atom_valence,
                                'atom_degree': OneHotAtomFeatures.atom_degree
                                }
                        )


                if not features:
                        features = list(node_featurers.keys())

                if onehot:
                        feature_matrix = list()

                else:
                        feature_matrix = np.zeros((mol.GetNumAtoms(), len(features)))

                for n, atom in enumerate(mol.GetAtoms()):
                    if onehot:
                        vector = []

                    for m, feat in enumerate(features):
                            parameter = node_featurers[feat](atom)

                            if onehot:
                                vector.append(parameter)
                                
                            else:
                                feature_matrix[n, m] = parameter

                    if onehot:
                        vector = np.hstack(vector)
                        feature_matrix.append(list(vector))


                if onehot:
                        feature_matrix = np.array(feature_matrix)

                if get_pos:
                       return feature_matrix, pos
        
                return feature_matrix



        @staticmethod
        def get_edge_list_chemical_topology(mol, undirected=True, dist=False, add_features=False, onehot=False):

            edge_index = list()
            edge_attr = list()

            encoder = OneHotAtomFeatures if onehot else StandardEncoding

            if dist:
                conf = mol.GetConformer()
                pos = conf.GetPositions()


            for bond in mol.GetBonds():
                connect = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]

                if dist:
                    r = pos[bond.GetBeginAtom().GetIdx()] - pos[bond.GetEndAtom().GetIdx()]
                    r_norm = np.linalg.norm(r)

                if add_features:
                    arom = encoder.aromaticity_bond(bond)
                    bond_type = encoder.bond_type(bond)
                    stereo = encoder.bond_stereo(bond)


                if undirected:
                    edge_index.extend([connect, connect[::-1]])

                    if add_features:
                        if dist:
                            bond_features = np.hstack([r_norm, arom, bond_type, stereo])
                        else:
                            bond_features = np.hstack([arom, bond_type, stereo])

                        edge_attr.append([bond_features, bond_features])


                else:
                    edge_index.append(connect)

                    if add_features:
                        if dist:
                            bond_features = np.hstack([r_norm, arom, bond_type, stereo])
                        else:
                            bond_features = np.hstack([arom, bond_type, stereo])
                            edge_attr.append(bond_features)
                        

            edge_index = np.array(edge_index).T

            if not np.count_nonzero(edge_index == 0) > 0:
                    edge_index -= 1

            if add_features:
                    return edge_index, np.vstack(edge_attr)
        
            return edge_index

        
        @staticmethod
        def get_edge_list_spatial_topology(mol: Chem.Mol, cutoff=5):

            conf = mol.GetConformer()
            pos = conf.GetPositions()

            r = np.linalg.norm(pos[:,np.newaxis] - pos[np.newaxis, :], axis=-1)
            edge_idx = np.array(np.nonzero((r <= cutoff) & (r != 0)))

            return edge_idx
