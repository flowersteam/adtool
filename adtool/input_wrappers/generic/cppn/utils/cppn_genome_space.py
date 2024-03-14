import os
from copy import deepcopy

import neat
from adtool.input_wrappers.generic.cppn import pytorchneat
from adtool.utils.spaces import BaseSpace


class CPPNGenomeSpace(BaseSpace):
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "../config.cfg")
        self.neat_config = neat.Config(
            pytorchneat.selfconnectiongenome.SelfConnectionGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        super().__init__(shape=None, dtype=None)

    def sample(self):
        genome = self.neat_config.genome_type(0)
        genome.configure_new(self.neat_config.genome_config)
        return genome

    def mutate(self, genome):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        new_genome = deepcopy(genome)
        new_genome.mutate(self.neat_config.genome_config)
        return new_genome

    def contains(self, x):
        # TODO
        return True

    def clamp(self, x):
        # TODO
        return x
