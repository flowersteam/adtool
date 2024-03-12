from copy import deepcopy
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

import neat
from auto_disc.auto_disc.maps.Map import Map
from auto_disc.legacy.input_wrappers.generic.cppn import pytorchneat
from auto_disc.utils.leaf.locators.locators import BlobLocator


class NEATParameterMap(Map):
    def __init__(
        self,
        premap_key: str = "genome",
        config_path: str = "./config.cfg",
        config_str: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key

        # config argument overrides config_path
        if not config_str:
            self.neat_config = neat.Config(
                pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
        else:
            with NamedTemporaryFile() as fp:
                # write string and then reset stream to beginning
                config_bytes = config_str.encode()
                fp.write(config_bytes)
                fp.seek(0)

                # initialize neat with temp file created
                config_path = fp.name
                self.neat_config = neat.Config(
                    pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                    neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,
                    neat.DefaultStagnation,
                    config_path,
                )

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)

        # check if either "genome" is not set or if we want to override
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            # overrides "genome" with new sample
            intermed_dict[self.premap_key] = self.sample()
        else:
            # passes "genome" through if it exists
            pass

        # also passes the neat_config needed to initialize the genome into
        # an initial state tensor
        intermed_dict["neat_config"] = self.neat_config

        return intermed_dict

    def sample(self) -> Any:
        """
        Samples a genome.
        """
        genome = self._sample_genome()
        return genome

    def mutate(self, dict: Dict) -> Dict:
        """
        Mutates the genome in the provided dict.
        """
        intermed_dict = deepcopy(dict)

        genome = intermed_dict[self.premap_key]
        genome.mutate(self.neat_config.genome_config)

        intermed_dict[self.premap_key] = genome

        return intermed_dict

    def _sample_genome(self) -> Any:
        genome = self.neat_config.genome_type(0)
        # randomly initializes the genome
        genome.configure_new(self.neat_config.genome_config)
        return genome
