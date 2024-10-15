import dataclasses
from copy import deepcopy
from typing import Dict
import random

from adtool.utils.leaf.Leaf import Leaf

import random

def mutate_rule(rule, min_signature=1, max_signature=4,max_value=4):
    """
    Mutate a single rule within the given signature range, ensuring a change always occurs.
    
    :param rule: A tuple representing a rule (e.g., (1, 2, 3))
    :param min_signature: Minimum signature (default: 1)
    :param max_signature: Maximum signature (default: 4)
    :return: A new, mutated rule
    """
    mutated_rule = rule.copy()

    

    

    while True:  # Keep trying until we make a change

        
        # Randomly choose a mutation operation
        mutation_type = random.choice(["change", "add", "remove"])

        
        if mutation_type == "change":
            # Change a single element
            index = random.randint(0, len(mutated_rule) - 1)
            new_value = random.randint(1, max_value+1)
            
            if new_value != mutated_rule[index]:
                max_value=max(max_value,new_value)
                mutated_rule[index] = new_value
                break
        
        elif mutation_type == "add" and len(mutated_rule) < max_signature:
            mutated_rule.append(random.randint(1, max_value+1))
            break
        
        elif mutation_type == "remove" and len(mutated_rule) > min_signature:
            # Remove a random element
            index = random.randint(0, len(mutated_rule) - 1)
            mutated_rule.pop(index)
            break
    
    return mutated_rule


def reduce_max(ruleset):
    # number are just tags, so use lower number
    # find number of distinct numbers
    numbers=set()
    for key in ["in", "out"]:
        for rule in ruleset[key]:
            for number in rule:
                numbers.add(number)
    numbers=list(numbers)
    correspondence={
        number: i+1 for i,number in enumerate(numbers)
    }
    # change the rules
    for key in ["in", "out"]:
        for rule in ruleset[key]:
            for i in range(len(rule)):
                rule[i]=correspondence[rule[i]]
    return ruleset

def mutate_ruleset(ruleset, min_signature=1, max_signature=4):
    """
    Mutate a ruleset, ensuring at least one rule is always changed.
    
    :param ruleset: A dictionary with 'in' and 'out' lists of rules
    :param min_signature: Minimum signature (default: 1)
    :param max_signature: Maximum signature (default: 4)
    :return: A new, mutated ruleset
    """
    mutated_ruleset = {
        "in": [],
        "out": []
    }
    
    changed = False

    max_value=max(max(max(ruleset["in"]),max(ruleset["out"])))

    for key in ["in", "out"]:
        for rule in ruleset[key]:
            
            if not changed or random.random() < 0.5:  # Ensure at least one change, then 50% chance for others
                mutated_rule = mutate_rule(rule, min_signature, max_signature,  max_value)
                mutated_ruleset[key].append(mutated_rule)
                if mutated_rule != rule:
                    changed = True
            else:
                mutated_ruleset[key].append(rule)
    
    # If no changes were made (very unlikely), force a change on a random rule
    if not changed:
        key = random.choice(["in", "out"])
        index = random.randint(0, len(mutated_ruleset[key]) - 1)
        mutated_ruleset[key][index] = mutate_rule(mutated_ruleset[key][index], min_signature, max_signature,  max_value)
    
    return reduce_max(mutated_ruleset)



class WPhysicsParameterMap(Leaf):
    def __init__(
        self,
        system,
        premap_key: str = "params",
        **config_decorator_kwargs,
    ):
        super().__init__()

        self.premap_key = premap_key
        self.init_rules = {
            "in": [[1, 1, 2], [3, 4, 1]],
            "out": [[1, 1, 4], [5, 4, 3], [2, 5, 1]]
        }

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:
        return {"dynamic_params": self.init_rules}

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)
        current_rules = parameter_dict["dynamic_params"]
        new_rules = mutate_ruleset(current_rules)

        print(f"Mutated {current_rules} to {new_rules}")

        intermed_dict["dynamic_params"] = new_rules
        return intermed_dict