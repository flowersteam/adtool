import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict

import torch
import numpy as np
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

import selfies as sf

import random


chemical_space=[('[#B-1]', 5.413716191342387e-05),
 ('[#B]', 0.00027068580956711934),
 ('[#Branch1]', 0.027068580956711934),
 ('[#Branch2]', 0.01624114857402716),
 ('[#Branch3]', 0.005413716191342387),
 ('[#C+1]', 5.413716191342387e-05),
 ('[#C-1]', 5.413716191342387e-05),
 ('[#C]', 0.0027068580956711935),
 ('[#N+1]', 0.00027068580956711934),
 ('[#N]', 0.005413716191342387),
 ('[#O+1]', 5.413716191342387e-05),
 ('[#P+1]', 2.7068580956711936e-05),
 ('[#P-1]', 2.7068580956711936e-05),
 ('[#P]', 5.413716191342387e-05),
 ('[#S+1]', 5.413716191342387e-05),
 ('[#S-1]', 5.413716191342387e-05),
 ('[#S]', 0.00027068580956711934),
 ('[=B+1]', 5.413716191342387e-06),
 ('[=B-1]', 5.413716191342387e-06),
 ('[=B]', 5.413716191342387e-05),
 ('[=Branch1]', 0.02165486476536955),
 ('[=Branch2]', 0.010827432382684774),
 ('[=Branch3]', 0.0027068580956711935),
 ('[=C+1]', 5.413716191342387e-05),
 ('[=C-1]', 5.413716191342387e-05),
 ('[=C]', 0.0812057428701358),
 ('[=N+1]', 0.00027068580956711934),
 ('[=N-1]', 0.00027068580956711934),
 ('[=N]', 0.01624114857402716),
 ('[=O+1]', 5.413716191342387e-05),
 ('[=O]', 0.05413716191342387),
 ('[=P+1]', 5.413716191342387e-06),
 ('[=P-1]', 5.413716191342387e-06),
 ('[=P]', 0.00027068580956711934),
 ('[=Ring1]', 0.0433097295307391),
 ('[=Ring2]', 0.027068580956711934),
 ('[=Ring3]', 0.010827432382684774),
 ('[=S+1]', 5.413716191342387e-05),
 ('[=S-1]', 5.413716191342387e-05),
 ('[=S]', 0.005413716191342387),
 ('[B+1]', 2.7068580956711936e-05),
 ('[B-1]', 2.7068580956711936e-05),
 ('[B]', 0.00027068580956711934),
 ('[Br]', 0.005413716191342387),
 ('[Branch1]', 0.05413716191342387),
 ('[Branch2]', 0.027068580956711934),
 ('[Branch3]', 0.010827432382684774),
 ('[C+1]', 0.00027068580956711934),
 ('[C-1]', 0.00027068580956711934),
 ('[C]', 0.13534290478355968),
 ('[Cl]', 0.010827432382684774),
 ('[F]', 0.00812057428701358),
 ('[H]', 0.1624114857402716),
 ('[I]', 0.0027068580956711935),
 ('[N+1]', 0.0027068580956711935),
 ('[N-1]', 0.0027068580956711935),
 ('[N]', 0.0433097295307391),
 ('[O+1]', 5.413716191342387e-05),
 ('[O-1]', 0.0027068580956711935),
 ('[O]', 0.0812057428701358),
 ('[P+1]', 5.413716191342387e-06),
 ('[P-1]', 5.413716191342387e-06),
 ('[P]', 0.0027068580956711935),
 ('[Ring1]', 0.05413716191342387),
 ('[Ring2]', 0.03248229714805432),
 ('[Ring3]', 0.01624114857402716),
 ('[S+1]', 0.00027068580956711934),
 ('[S-1]', 0.00027068580956711934),
 ('[S]', 0.010827432382684774)]



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
        
        while True:
            random_selfies=""
            for _ in range(random.randint(10, 200)):
                random_selfies += np.random.choice( 
                    [x[0] for x in chemical_space], p=[x[1] for x in chemical_space]
                )
            smiles=sf.decoder(random_selfies)
            if self.minimal_checks(smiles):
                break



        p_dict = {
            "dynamic_params": {
                "smiles": smiles
            
            }
        }
        return p_dict
    
    def minimal_checks(self, smiles: str) -> bool:


        # convert to selfies
        selfies = sf.encoder(smiles)
        if len(list(sf.split_selfies(selfies))) > 2 :
            return True
        return False

    def mutate(self, parameter_dict: Dict) -> Dict:
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
