from random import gauss, random, uniform

import neat.attributes
import neat.genes
import neat.genome


class ConnectionWeightAttribute(neat.attributes.BaseAttribute):
    """
    Class for connection weight attributes. Is introduced to allow a differentiation between self connections of a node and connections to other nodes.
    """

    _config_items = {
        "other_init_mean": [float, None],
        "other_init_stdev": [float, None],
        "other_init_type": [str, "gaussian"],
        "other_replace_rate": [float, None],
        "other_mutate_rate": [float, None],
        "other_mutate_power": [float, None],
        "other_max_value": [float, None],
        "other_min_value": [float, None],
        "self_init_mean": [float, None],
        "self_init_stdev": [float, None],
        "self_init_type": [str, "gaussian"],
        "self_replace_rate": [float, None],
        "self_mutate_rate": [float, None],
        "self_mutate_power": [float, None],
        "self_max_value": [float, None],
        "self_min_value": [float, None],
    }

    def clamp(self, value, is_other_connect, config):
        if is_other_connect:
            min_value = getattr(config, self.other_min_value_name)
            max_value = getattr(config, self.other_max_value_name)
        else:
            min_value = getattr(config, self.self_min_value_name)
            max_value = getattr(config, self.self_max_value_name)

        return max(min(value, max_value), min_value)

    def init_value(self, is_other_connect, config):
        if is_other_connect:
            mean = getattr(config, self.other_init_mean_name)
            stdev = getattr(config, self.other_init_stdev_name)
            init_type = getattr(config, self.other_init_type_name).lower()
        else:
            mean = getattr(config, self.self_init_mean_name)
            stdev = getattr(config, self.self_init_stdev_name)
            init_type = getattr(config, self.self_init_type_name).lower()

        if ("gauss" in init_type) or ("normal" in init_type):
            return self.clamp(gauss(mean, stdev), is_other_connect, config)

        if "uniform" in init_type:
            if is_other_connect:
                min_value = getattr(config, self.other_min_value_name)
                max_value = getattr(config, self.other_max_value_name)
            else:
                min_value = getattr(config, self.self_min_value_name)
                max_value = getattr(config, self.self_max_value_name)

            min_value = max(min_value, (mean - (2 * stdev)))
            max_value = min(max_value, (mean + (2 * stdev)))

            return uniform(min_value, max_value)

        raise RuntimeError("Unknown init_type {!r}!".format(init_type))

    def mutate_value(self, value, is_other_connect, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency

        if is_other_connect:
            mutate_rate = getattr(config, self.other_mutate_rate_name)
        else:
            mutate_rate = getattr(config, self.self_mutate_rate_name)

        r = random()
        if r < mutate_rate:
            if is_other_connect:
                mutate_power = getattr(config, self.other_mutate_power_name)
            else:
                mutate_power = getattr(config, self.self_mutate_power_name)

            return self.clamp(
                value + gauss(0.0, mutate_power), is_other_connect, config
            )

        if is_other_connect:
            replace_rate = getattr(config, self.other_replace_rate_name)
        else:
            replace_rate = getattr(config, self.self_replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(is_other_connect, config)

        return value

    def validate(self, config):  # pragma: no cover
        pass


class SelfConnectionGene(neat.genes.BaseGene):
    _gene_attributes = [
        ConnectionWeightAttribute("weight"),
        neat.attributes.BoolAttribute("enabled"),
    ]

    def __init__(self, key):
        assert isinstance(
            key, tuple
        ), "SelfConnectionGene key must be a tuple, not {!r}".format(key)
        neat.genes.BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient

    def init_attributes(self, config):
        for a in self._gene_attributes:
            if isinstance(a, ConnectionWeightAttribute):
                # check if the connection is between two different nodes or the same node
                # the key holds the two IDs of both nodes: key = (id_node_1, id_node_2)
                is_other_connect = self.key[0] is not self.key[1]
                val = a.init_value(is_other_connect, config)
            else:
                val = a.init_value(config)

            setattr(self, a.name, val)

    def mutate(self, config):
        for a in self._gene_attributes:
            old_val = getattr(self, a.name)

            if isinstance(a, ConnectionWeightAttribute):
                # check if the connection is between two different nodes or the same node
                # the key holds the two IDs of both nodes: key = (id_node_1, id_node_2)
                is_other_connect = self.key[0] is not self.key[1]
                new_val = a.mutate_value(old_val, is_other_connect, config)
            else:
                new_val = a.mutate_value(old_val, config)

            setattr(self, a.name, new_val)


class SelfConnectionGenome(neat.genome.DefaultGenome):
    """
    A genome for generalized neural networks. Derived from neat.DefaultGenome.
    Extends the default genome to allow specific configuration for self connections of neurons.

    Uses the SelfConnectionGene class instead of the neat.genes.DefaultConnectionGene class for connections.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = neat.genes.DefaultNodeGene
        param_dict["connection_gene_type"] = SelfConnectionGene
        return neat.genome.DefaultGenomeConfig(param_dict)
