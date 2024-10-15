import typing
from copy import deepcopy
import numpy as np
import networkx as nx
from addict import Dict
from adtool.wrappers.BoxProjector import BoxProjector
from adtool.utils.leaf.Leaf import Leaf

class WPhysicsStatistics(Leaf):
    def __init__(
        self,
        system,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()

        self.max_steps = system.max_steps
        self.max_entries = system.max_entries
        self.premap_key = premap_key
        self.postmap_key = postmap_key


        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: typing.Dict) -> typing.Dict:
        intermed_dict = deepcopy(input)

        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = intermed_dict[self.premap_key]
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(intermed_dict[raw_output_key])

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        goal= self.projector.sample()
        # optimize for long flights
        goal[0]=1
        return goal

    def _calc_static_statistics(self, output: Dict) -> np.ndarray:
        nb_steps, edges = output["steps"], output["final_state"]
        
        G = nx.MultiDiGraph()
        for edge in edges:
            if len(edge) == 1:
                G.add_node(edge[0])
            else:
                for i in range(len(edge) - 1):
                    G.add_edge(edge[i], edge[i + 1])
        
        flight = nb_steps / self.max_steps
        nodes_edges = 1 / (1 + G.number_of_nodes() / G.number_of_edges())
        avg_degree = sum(d for n, d in G.degree()) / G.number_of_nodes()
        density = nx.density(G)
        max_degree_centrality = max(nx.degree_centrality(G).values())
        min_degree_centrality = min(nx.degree_centrality(G).values())

        max_degree = max([d for n, d in G.degree()])
        min_degree = min([d for n, d in G.degree()])

        load=sum([len(x) for x in edges]) / self.max_entries

        return np.array([flight, nodes_edges, avg_degree,

                            min_degree/max_degree
        
        , density, max_degree_centrality,
                         min_degree_centrality/max_degree_centrality ,load])