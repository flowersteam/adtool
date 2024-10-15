import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import io
import networkx as nx
import matplotlib.pyplot as plt
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import time
from networkx.drawing.nx_agraph import graphviz_layout

def pattern2string(pattern):
    return str(pattern).replace("[", "{").replace("]", "}").replace("(", "{").replace(")", "}")

def rules2string(rules):
    # write rule to this format {{{1, 1, 2}, {3, 2, 4}} -> {{2, 2, 4}, {5, 3, 2}, {3, 1, 3}}}
    rules_str = str(rules["in"]) + " -> " + str(rules["out"])
    rules_str = pattern2string(rules_str)
    return rules_str



def initial_condition(rules):
    print(rules)
    in_rules=rules["in"]
    #fill with 0
    init=[]
    for i in range(len(in_rules)):
        init.append([1]*len(in_rules[i]))

    return init


class WPhysics:
    def __init__(self, max_steps: int = 200,max_entries: int = 400) -> None:
        self.max_steps = max_steps
        self.max_entries = max_entries
        self.params = None
        self.final_state = None
        self.steps = None
        self.session = WolframLanguageSession('/usr/local/Wolfram/Wolfram/14.1/Executables/WolframKernel')


    def __del__(self):
        self.session.terminate()

    def run_simulation(self):
        current_state = initial_condition(self.params['dynamic_params'])
        rules_str = rules2string(self.params['dynamic_params'])

        for i in range(self.max_steps):
            past_time = time.time()

            expr = f'ResourceFunction["WolframModel"][{{{rules_str}}}, {pattern2string(current_state)}, 1, "FinalState"]'
            result = self.session.evaluate(wlexpr(expr))

            result = [list(x) for x in result]

            if result == current_state or (i > 0 and time.time() - past_time > 1) or sum([len(x) for x in result]) > self.max_entries:
                break

            current_state = result

        self.steps = i + 1
        self.final_state = current_state

    def map(self, input: Dict) -> Dict:
        self.params = input["params"]
        self.run_simulation()
        input["output"] = {
            "final_state": self.final_state,
            "steps": self.steps
        }
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        G = nx.MultiDiGraph()
        print("final state", self.final_state)
        for edge in self.final_state:
            if len(edge) == 1:
                # add an unary node
                G.add_node(edge[0])
            else:
                for i in range(len(edge) - 1):
                    G.add_edge(edge[i], edge[i + 1], color='red')



        fig, ax = plt.subplots(figsize=(3, 3))

        ax.set_axis_off()



        num_nodes = G.number_of_nodes()


        if num_nodes <15:

            graph_ax = fig.add_axes([0.3, 0.3, 0.4, 0.4])

            graph_ax.set_axis_off()
        else:
            # no axes
            graph_ax = None



# Remove axis from the graph subplot


# Adjust the layout to prevent clipping


        pos = graphviz_layout(G, prog="sfdp")

        nx.draw(G, pos,
                ax=graph_ax, with_labels=False,
                node_size=20 )
        plt.tight_layout()
     #   edge_labels = {(u, v): f'{u}->{v}' for u, v in G.edges()}
       # nx.draw_networkx_edge_labels(G, pos)

        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        plt.close()

        return [(buf.getvalue(), "png")]