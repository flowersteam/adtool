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


chemical_space=[('[#B-1]', 6.463455621913702e-05),
 ('[#B]', 0.00032317278109568514),
 ('[#Branch1]', 0.03231727810956851),
 ('[#Branch2]', 0.019390366865741106),
 ('[#Branch3]', 0.006463455621913703),
 ('[#C+1]', 6.463455621913702e-05),
 ('[#C-1]', 6.463455621913702e-05),
 ('[#C]', 0.0032317278109568514),
 ('[#N+1]', 0.00032317278109568514),
 ('[#N]', 0.006463455621913703),
 ('[#O+1]', 6.463455621913702e-05),
 ('[#P+1]', 3.231727810956851e-05),
 ('[#P-1]', 3.231727810956851e-05),
 ('[#P]', 6.463455621913702e-05),
 ('[#S+1]', 6.463455621913702e-05),
 ('[#S-1]', 6.463455621913702e-05),
 ('[#S]', 0.00032317278109568514),
 ('[=B+1]', 6.463455621913703e-06),
 ('[=B-1]', 6.463455621913703e-06),
 ('[=B]', 6.463455621913702e-05),
 ('[=Branch1]', 0.02585382248765481),
 ('[=Branch2]', 0.012926911243827405),
 ('[=Branch3]', 0.0032317278109568514),
 ('[=C+1]', 6.463455621913702e-05),
 ('[=C-1]', 6.463455621913702e-05),
 ('[=C]', 0.09695183432870554),
 ('[=N+1]', 0.00032317278109568514),
 ('[=N-1]', 0.00032317278109568514),
 ('[=N]', 0.019390366865741106),
 ('[=O+1]', 6.463455621913702e-05),
 ('[=O]', 0.06463455621913702),
 ('[=P+1]', 6.463455621913703e-06),
 ('[=P-1]', 6.463455621913703e-06),
 ('[=P]', 0.00032317278109568514),
 ('[=Ring1]', 0.05170764497530962),
 ('[=Ring2]', 0.03231727810956851),
 ('[=Ring3]', 0.012926911243827405),
 ('[=S+1]', 6.463455621913702e-05),
 ('[=S-1]', 6.463455621913702e-05),
 ('[=S]', 0.006463455621913703),
 ('[B+1]', 3.231727810956851e-05),
 ('[B-1]', 3.231727810956851e-05),
 ('[B]', 0.00032317278109568514),
 ('[Br]', 0.006463455621913703),
 ('[Branch1]', 0.06463455621913702),
 ('[Branch2]', 0.03231727810956851),
 ('[Branch3]', 0.012926911243827405),
 ('[C+1]', 0.00032317278109568514),
 ('[C-1]', 0.00032317278109568514),
 ('[C]', 0.16158639054784255),
 ('[Cl]', 0.012926911243827405),
 ('[F]', 0.009695183432870553),
 ('[I]', 0.0032317278109568514),
 ('[N+1]', 0.0032317278109568514),
 ('[N-1]', 0.0032317278109568514),
 ('[N]', 0.05170764497530962),
 ('[O+1]', 6.463455621913702e-05),
 ('[O-1]', 0.0032317278109568514),
 ('[O]', 0.09695183432870554),
 ('[P+1]', 6.463455621913703e-06),
 ('[P-1]', 6.463455621913703e-06),
 ('[P]', 0.0032317278109568514),
 ('[Ring1]', 0.06463455621913702),
 ('[Ring2]', 0.03878073373148221),
 ('[Ring3]', 0.019390366865741106),
 ('[S+1]', 0.00032317278109568514),
 ('[S-1]', 0.00032317278109568514),
 ('[S]', 0.012926911243827405)]



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

        # Lipinski's rule of five for example


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
