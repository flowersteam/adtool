import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict

import torch
from crem.crem import mutate_mol, grow_mol, link_mols

import numpy as np
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

import selfies as sf

from rdkit.Chem import Lipinski
from rdkit import Chem
from rdkit.Chem import AllChem

import random


fragments_db="examples/docking/maps/replacements02_sa2.db"

chemical_space=[('[=B+1]', 5.747588886462131e-06),
 ('[=B-1]', 5.747588886462131e-06),
 ('[=B]', 5.747588886462131e-05),
 ('[=Branch1]', 0.022990355545848523),
 ('[=Branch2]', 0.011495177772924262),
 ('[=Branch3]', 0.0028737944432310654),
 ('[=C+1]', 5.747588886462131e-05),
 ('[=C-1]', 5.747588886462131e-05),
 ('[=C]', 0.08621383329693195),
 ('[=N+1]', 0.00028737944432310656),
 ('[=N-1]', 0.00028737944432310656),
 ('[=N]', 0.017242766659386392),
 ('[=O+1]', 5.747588886462131e-05),
 ('[=O]', 0.05747588886462131),
 ('[=P+1]', 5.747588886462131e-06),
 ('[=P-1]', 5.747588886462131e-06),
 ('[=P]', 0.00028737944432310656),
 ('[=Ring1]', 0.045980711091697046),
 ('[=Ring2]', 0.028737944432310654),
 ('[=Ring3]', 0.011495177772924262),
 ('[=S+1]', 5.747588886462131e-05),
 ('[=S-1]', 5.747588886462131e-05),
 ('[=S]', 0.005747588886462131),
 ('[B+1]', 2.8737944432310656e-05),
 ('[B-1]', 2.8737944432310656e-05),
 ('[B]', 0.00028737944432310656),
 ('[Br]', 0.005747588886462131),
 ('[Branch1]', 0.05747588886462131),
 ('[Branch2]', 0.028737944432310654),
 ('[Branch3]', 0.011495177772924262),
 ('[C+1]', 0.00028737944432310656),
 ('[C-1]', 0.00028737944432310656),
 ('[C]', 0.14368972216155326),
 ('[Cl]', 0.011495177772924262),
 ('[F]', 0.008621383329693196),
 ('[H]', 0.1724276665938639),
 ('[I]', 0.0028737944432310654),
 ('[N+1]', 0.0028737944432310654),
 ('[N-1]', 0.0028737944432310654),
 ('[N]', 0.045980711091697046),
 ('[O+1]', 5.747588886462131e-05),
 ('[O-1]', 0.0028737944432310654),
 ('[O]', 0.08621383329693195),
 ('[P+1]', 5.747588886462131e-06),
 ('[P-1]', 5.747588886462131e-06),
 ('[P]', 0.0028737944432310654),
 ('[Ring1]', 0.05747588886462131),
 ('[Ring2]', 0.034485533318772785),
 ('[Ring3]', 0.017242766659386392),
 ('[S+1]', 0.00028737944432310656),
 ('[S-1]', 0.00028737944432310656),
 ('[S]', 0.011495177772924262)]


from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors

class SmilesError(Exception): pass

def log_partition_coefficient(smiles):
    '''
    Returns the octanol-water partition coefficient given a molecule SMILES 
    string
    '''
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        raise SmilesError('%s returns a None molecule' % smiles)
        
    return Crippen.MolLogP(mol)
    
def lipinski_trial(smiles):
    '''
    Returns which of Lipinski's rules a molecule has failed, or an empty list
    
    Lipinski's rules are:
    Hydrogen bond donors <= 5
    Hydrogen bond acceptors <= 10
    Molecular weight < 500 daltons
    logP < 5
    '''

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Exception('%s is not a valid SMILES string' % smiles)
    
    
    num_hdonors = Lipinski.NumHDonors(mol)
    

    
    if num_hdonors > 5:
        return False
    
    num_hacceptors = Lipinski.NumHAcceptors(mol)
   
    if num_hacceptors > 10:
        return False
    
    mol_weight = Descriptors.MolWt(mol)
        
    if mol_weight >= 500:
        return False
        
    mol_logp = Crippen.MolLogP(mol)

    if mol_logp >= 5:
        return False
    
    try:
        AllChem.EmbedMolecule(mol)
    except:
        return False
    
    return True


class DockingParameterMap(Leaf):
    def __init__(
        self,
        system,
        premap_key: str = "params",
        **config_decorator_kwargs,
    ):
        super().__init__()



        self.premap_key = premap_key
        self.seed_smiles = config_decorator_kwargs.get("seed_smiles", "CC1=CC(=O)C(=C(C1=O)O)O")



    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:
        # sample from chemical space with probabilities
        
        # while True:
        #     random_selfies=""
        #     for _ in range(random.randint(10, 200)):
        #         random_selfies += np.random.choice( 
        #             [x[0] for x in chemical_space], p=[x[1] for x in chemical_space]
        #         )
        #     smiles=sf.decoder(random_selfies)
        #     if self.minimal_checks(smiles):
        #         break

        smiles = self.seed_smiles



        p_dict = {
            "dynamic_params": {
                "smiles": smiles
            
            }
        }
        return p_dict
    
    def minimal_checks(self, smiles: str) -> bool:

        # Lipinski's rule of five for example        
        if not lipinski_trial(smiles):
            return False


        # # convert to selfies
        # selfies = sf.encoder(smiles)
        # if len(list(sf.split_selfies(selfies))) > 2 :
        #     return True
        return True

    def old_mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)
        current_smiles = parameter_dict["dynamic_params"]["smiles"]
        selfies = sf.encoder(current_smiles)
        list_selfies=list(sf.split_selfies(selfies))
        
        # remove from 0 to 2 elements
        # nb_remove = np.random.randint(0, 3)
        # for _ in range(nb_remove):
        #     list_selfies.pop(np.random.randint(0, len(list_selfies)))

        while True:
            for _ in range(
            #    nb_remove == 0,
                np.random.randint(1
                                # + nb_remove==0,
                                , 3)):
                list_selfies.insert(np.random.randint(0, len(list_selfies)),
                                    np.random.choice(
                                        [x[0] for x in chemical_space],
                                        p=[x[1] for x in chemical_space]
                                    )
                )

            mutated_selfies = "".join(list_selfies)
            mutated_smiles = sf.decoder(mutated_selfies)
            if self.minimal_checks(mutated_smiles) and mutated_smiles!=current_smiles:
                break
            
        

            

        print(f"Mutated {current_smiles} to {mutated_smiles}")

        intermed_dict["dynamic_params"] = {
            "smiles": mutated_smiles
        }
        return intermed_dict


    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)
        current_smiles = parameter_dict["dynamic_params"]["smiles"]

        print("current_smiles", current_smiles)
        
        mol=Chem.MolFromSmiles(current_smiles)
        # count number of atoms
        num_atoms = mol.GetNumAtoms()
        if 100<num_atoms  or num_atoms<2:
            new_smiles=  next(grow_mol(Chem.AddHs(mol), db_name=fragments_db))
            
        else:
            # randomly augment or mutate
            if random.random() < 0.5:
                new_smiles = next(mutate_mol(mol, db_name=fragments_db))
            else:
                new_smiles = next(grow_mol(mol, db_name=fragments_db))

        
        # convert to canonical smiles
        new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(new_smiles))

        print(f"Mutated {current_smiles} to {new_smiles}")

        intermed_dict["dynamic_params"] = {
            "smiles": new_smiles
        }
        return intermed_dict