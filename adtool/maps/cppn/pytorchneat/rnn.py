import copy

import graphviz
import torch
from adtool.input_wrappers.generic.cppn.pytorchneat.activations import (
    str_to_activation,
)
from adtool.input_wrappers.generic.cppn.pytorchneat.aggregations import (
    str_to_aggregation,
)
from torch import nn

torch.set_default_dtype(torch.float64)

use_Minkowski_inputs = False
if use_Minkowski_inputs:
    import MinkowskiEngine as ME


class RecurrentNetwork(nn.Module):
    def __init__(
        self,
        input_neuron_ids,
        hidden_neuron_ids,
        output_neuron_ids,
        biases,
        responses,
        activations,
        connections,
        device="cpu",
    ):
        """
        :param input_neuron_ids: list of ids of input neurons
        :param hidden_neuron_ids: list of ids of hidden neurons
        :param output_neuron_ids: list of ids of output neurons
        """
        super().__init__()
        self.device = device

        self.input_key_to_idx = {k: i for i, k in enumerate(input_neuron_ids)}
        self.n_inputs = len(input_neuron_ids)
        self.hidden_key_to_idx = {k: i for i, k in enumerate(hidden_neuron_ids)}
        self.n_outputs = len(output_neuron_ids)
        self.output_key_to_idx = {k: i for i, k in enumerate(output_neuron_ids)}
        self.n_hidden = len(hidden_neuron_ids)

        # connection parameters
        input_to_hidden = torch.zeros((self.n_inputs, self.n_hidden))
        input_to_hidden_mask = torch.zeros(
            (self.n_inputs, self.n_hidden), dtype=torch.bool
        )
        hidden_to_hidden = torch.zeros((self.n_hidden, self.n_hidden))
        hidden_to_hidden_mask = torch.zeros(
            (self.n_hidden, self.n_hidden), dtype=torch.bool
        )
        output_to_hidden = torch.zeros((self.n_outputs, self.n_hidden))
        output_to_hidden_mask = torch.zeros(
            (self.n_outputs, self.n_hidden), dtype=torch.bool
        )
        input_to_output = torch.zeros((self.n_inputs, self.n_outputs))
        input_to_output_mask = torch.zeros(
            (self.n_inputs, self.n_outputs), dtype=torch.bool
        )
        hidden_to_output = torch.zeros((self.n_hidden, self.n_outputs))
        hidden_to_output_mask = torch.zeros(
            (self.n_hidden, self.n_outputs), dtype=torch.bool
        )
        output_to_output = torch.zeros((self.n_outputs, self.n_outputs))
        output_to_output_mask = torch.zeros(
            (self.n_outputs, self.n_outputs), dtype=torch.bool
        )
        for k, v in connections.items():
            if k[1] in hidden_neuron_ids:
                if k[0] in input_neuron_ids:
                    input_to_hidden[
                        self.input_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]
                    ] = v
                    input_to_hidden_mask[
                        self.input_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]
                    ] = True
                elif k[0] in hidden_neuron_ids:
                    hidden_to_hidden[
                        self.hidden_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]
                    ] = v
                    hidden_to_hidden_mask[
                        self.hidden_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]
                    ] = True
                elif k[0] in output_neuron_ids:
                    output_to_hidden[
                        self.output_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]
                    ] = v
                    output_to_hidden_mask[
                        self.output_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]
                    ] = True
            elif k[1] in output_neuron_ids:
                if k[0] in input_neuron_ids:
                    input_to_output[
                        self.input_key_to_idx[k[0]], self.output_key_to_idx[k[1]]
                    ] = v
                    input_to_output_mask[
                        self.input_key_to_idx[k[0]], self.output_key_to_idx[k[1]]
                    ] = True
                elif k[0] in hidden_neuron_ids:
                    hidden_to_output[
                        self.hidden_key_to_idx[k[0]], self.output_key_to_idx[k[1]]
                    ] = v
                    hidden_to_output_mask[
                        self.hidden_key_to_idx[k[0]], self.output_key_to_idx[k[1]]
                    ] = True
                elif k[0] in output_neuron_ids:
                    output_to_output[
                        self.output_key_to_idx[k[0]], self.output_key_to_idx[k[1]]
                    ] = v
                    output_to_output_mask[
                        self.output_key_to_idx[k[0]], self.output_key_to_idx[k[1]]
                    ] = True
        self.register_parameter("input_to_hidden", nn.Parameter(input_to_hidden))
        self.register_buffer("input_to_hidden_mask", input_to_hidden_mask)
        self.register_parameter("hidden_to_hidden", nn.Parameter(hidden_to_hidden))
        self.register_buffer("hidden_to_hidden_mask", hidden_to_hidden_mask)
        self.register_parameter("output_to_hidden", nn.Parameter(output_to_hidden))
        self.register_buffer("output_to_hidden_mask", output_to_hidden_mask)
        self.register_parameter("input_to_output", nn.Parameter(input_to_output))
        self.register_buffer("input_to_output_mask", input_to_output_mask)
        self.register_parameter("hidden_to_output", nn.Parameter(hidden_to_output))
        self.register_buffer("hidden_to_output_mask", hidden_to_output_mask)
        self.register_parameter("output_to_output", nn.Parameter(output_to_output))
        self.register_buffer("output_to_output_mask", output_to_output_mask)

        # mask nodes that does not have incoming connections (their bias should not be computed and influence the rest)
        hidden_mask = (
            self.input_to_hidden_mask.sum(0)
            + self.hidden_to_hidden_mask.sum(0)
            + self.output_to_hidden_mask.sum(0)
        ) > 0
        self.register_buffer("hidden_mask", hidden_mask)
        output_mask = (
            self.input_to_output_mask.sum(0)
            + self.hidden_to_output_mask.sum(0)
            + self.output_to_output_mask.sum(0)
        ) > 0
        self.register_buffer("output_mask", output_mask)

        # node parameters
        # biases
        self.hidden_biases = []
        self.output_biases = []
        for k, v in biases.items():
            if k in hidden_neuron_ids:
                self.hidden_biases.append(v)
            elif k in output_neuron_ids:
                self.output_biases.append(v)
        self.hidden_biases = nn.Parameter(torch.tensor(self.hidden_biases))
        self.output_biases = nn.Parameter(torch.tensor(self.output_biases))
        # responses
        self.hidden_responses = []
        self.output_responses = []
        for k, v in responses.items():
            if k in hidden_neuron_ids:
                self.hidden_responses.append(v)
            elif k in output_neuron_ids:
                self.output_responses.append(v)
        self.hidden_responses = nn.Parameter(torch.tensor(self.hidden_responses))
        self.output_responses = nn.Parameter(torch.tensor(self.output_responses))

        # activations
        self.hidden_activations = []
        self.output_activations = []
        for k, v in activations.items():
            if k in hidden_neuron_ids:
                self.hidden_activations.append(v)
            elif k in output_neuron_ids:
                self.output_activations.append(v)

        # push to device
        self.to(self.device)

    def forward(self, inputs):
        """
        :param inputs: tensor of size (batch_size, n_inputs)
        Note: output of a node as follows: activation(bias+(response∗aggregation(inputs)))
        returns: (batch_size, n_outputs)
        """
        batch_size = len(inputs)
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        after_pass_hidden_activs = torch.zeros_like(self.hidden_activs.detach()).to(
            self.device
        )
        after_pass_output_activs = torch.zeros_like(self.output_activs.detach()).to(
            self.device
        )

        # mask connection in case sparse 0 values have been changed during gradient descent
        self.input_to_hidden.data *= self.input_to_hidden_mask
        self.hidden_to_hidden.data *= self.hidden_to_hidden_mask
        self.output_to_hidden.data *= self.output_to_hidden_mask
        self.input_to_output.data *= self.input_to_output_mask
        self.hidden_to_output.data *= self.hidden_to_output_mask
        self.output_to_output.data *= self.output_to_output_mask

        # Step 1: multiply incoming inputs with connection weights
        after_pass_hidden_activs = (
            inputs.mm(self.input_to_hidden)
            + self.hidden_activs.mm(self.hidden_to_hidden)
            + self.output_activs.mm(self.output_to_hidden)
        )  # batch_size, n_hidden

        after_pass_output_activs = (
            inputs.mm(self.input_to_output)
            + self.hidden_activs.mm(self.hidden_to_output)
            + self.output_activs.mm(self.output_to_output)
        )  # batch_size, n_outputs

        # Step 2: x = agg(x)
        # here sum aggregation already happened in Step 1 within torch.mm

        # Step 3: x = bias + reponse * x
        after_pass_hidden_activs = (
            self.hidden_biases + self.hidden_responses * after_pass_hidden_activs
        )
        after_pass_output_activs = (
            self.output_biases + self.output_responses * after_pass_output_activs
        )

        # Step 3: x = act(x)
        for hidden_neuron_idx in range(self.n_hidden):
            after_pass_hidden_activs[:, hidden_neuron_idx] = self.hidden_activations[
                hidden_neuron_idx
            ](after_pass_hidden_activs[:, hidden_neuron_idx].clone())
        for output_neuron_idx in range(self.n_outputs):
            after_pass_output_activs[:, output_neuron_idx] = self.output_activations[
                output_neuron_idx
            ](after_pass_output_activs[:, output_neuron_idx].clone())

        self.hidden_activs = self.hidden_mask * after_pass_hidden_activs
        self.output_activs = self.output_mask * after_pass_output_activs

        return self.output_activs

    def activate(self, inputs, n_passes=1):
        """
        :param inputs: tensor of size (..., n_inputs) => only the last dim matter, then CPPN is applied on all coordinates
        :param n_passes: number of passes in the RNN TODO: currently global passes should it be internal passes as in pytorch-neat by uber research ?
        returns: (..., n_outputs)
        """
        input_size = inputs.shape[:-1]
        batch_size = torch.prod(torch.tensor(input_size))
        assert inputs.shape[-1] == self.n_inputs

        if isinstance(inputs, torch.Tensor):
            forward_inputs = inputs.view(-1, self.n_inputs)
        elif use_Minkowski_inputs:
            forward_inputs = inputs.features

        self.hidden_activs = torch.zeros((batch_size, self.n_hidden)).to(self.device)
        self.output_activs = torch.zeros((batch_size, self.n_outputs)).to(self.device)

        for _ in range(n_passes):
            outputs = self.forward(forward_inputs)

        if isinstance(inputs, torch.Tensor):
            outputs = outputs.view(input_size + (self.n_outputs,))
        elif use_Minkowski_inputs:
            outputs = ME.SparseTensor(
                outputs,
                coordinate_map_key=inputs.coordinate_map_key,
                coordinate_manager=inputs.coordinate_manager,
            )

        return outputs

    def draw_net(
        self, view=False, filename=None, node_names=None, node_colors=None, fmt="svg"
    ):
        if node_names is None:
            node_names = {}

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            "shape": "circle",
            "fontsize": "9",
            "height": "0.2",
            "width": "0.2",
        }

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        for k, idx in self.input_key_to_idx.items():
            name = node_names.get(k, str(k))
            input_attrs = {
                "style": "filled",
                "shape": "box",
                "fillcolor": node_colors.get(k, "lightgray"),
            }
            dot.node(name, _attributes=input_attrs)

        for k, idx in self.output_key_to_idx.items():
            name = node_names.get(k, str(k))
            node_attrs = {
                "style": "filled",
                "shape": "box",
                "fillcolor": node_colors.get(k, "lightblue"),
                "fontsize": "9",
                "fontcolor": node_colors.get(k, "blue"),
                "xlabel": f"{self.output_activations[idx].__name__[:-11]}({self.output_biases[idx]:.1f}+\\n{self.output_responses[idx]:.1f}*sum(inputs))",
            }

            dot.node(name, _attributes=node_attrs)

        for k, idx in self.hidden_key_to_idx.items():
            attrs = {
                "style": "filled",
                "fillcolor": node_colors.get(n, "white"),
                "fontsize": "9",
                "xlabel": f"{self.hidden_activations[idx].__name__[:-11]}({self.hidden_biases[idx]:.1f}+\\n{self.hidden_responses[idx]:.1f}*sum(inputs))",
            }

            dot.node(str(k), _attributes=attrs)

        connections = {}
        for from_k, to_k in torch.cat(
            [
                torch.stack(torch.where(self.input_to_hidden)).t(),
                torch.stack(torch.where(self.hidden_to_hidden)).t(),
                torch.stack(torch.where(self.output_to_output)).t(),
                torch.stack(torch.where(self.input_to_output)).t(),
                torch.stack(torch.where(self.hidden_to_output)).t(),
                torch.stack(torch.where(self.output_to_output)).t(),
            ]
        ):
            connections[(from_k.item(), to_k.item())] = self.input_to_hidden[
                from_k, to_k
            ]
        for cg, cg_weight in connections.items():
            input = cg[0]
            output = cg[1]
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid"
            color = "green" if cg_weight > 0 else "red"
            width = str(abs(cg_weight))
            dot.edge(
                a,
                b,
                _attributes={
                    "style": style,
                    "color": color,
                    "fontcolor": color,
                    "penwidth": width,
                    "fontsize": "9",
                    "label": f"w={cg_weight:.1f}",
                },
            )

        dot.render(filename, view=view)

        return dot

    @staticmethod
    def create(genome, config, device="cpu"):
        """Receives a genome and returns the RecurrentCPPN()"""
        genome_config = config.genome_config

        input_neuron_ids = genome_config.input_keys  # list of ids of input neurons
        output_neuron_ids = genome_config.output_keys  # list of ids of output neurons
        hidden_neuron_ids = list(genome.nodes.keys())[
            len(output_neuron_ids) :
        ]  # list of ids of hidden neurons (neither input nor ouput)

        # biases = {node_id: nodes[node_id].bias}
        biases = dict.fromkeys(genome.nodes.keys())
        # responses = {node_id: nodes[node_id].response}
        responses = dict.fromkeys(genome.nodes.keys())
        # activations = {node_id: nodes[node_id].activation}
        activations = dict.fromkeys(genome.nodes.keys())
        # aggregations = {node_id: nodes[node_id].aggregation}
        aggregations = dict.fromkeys(genome.nodes.keys())

        for node_id, node in genome.nodes.items():
            biases[node_id] = node.bias
            responses[node_id] = node.response
            activations[node_id] = str_to_activation[node.activation]
            assert (
                node.aggregation == "sum"
            ), "this class only implements sum aggregation, please use RecurrentNetworkAgg for other types of aggregations"
            aggregations[node_id] = str_to_aggregation[node.aggregation]

        connections = dict.fromkeys(
            genome.connections.keys()
        )  # connections = {(from_node_id, to_node_id): connections[(from_node_id, to_node_id)].weight}
        for (from_id, to_id), connection in genome.connections.items():
            if connection.enabled:
                connections[(from_id, to_id)] = connection.weight
            else:
                del connections[(from_id, to_id)]

        return RecurrentNetwork(
            input_neuron_ids,
            hidden_neuron_ids,
            output_neuron_ids,
            biases,
            responses,
            activations,
            connections,
            device=device,
        )

    def update_genome(self, genome):
        for k, v in genome.nodes.items():
            if k in self.output_key_to_idx.keys():
                v.bias = self.output_biases[self.output_key_to_idx[k]].data
                v.responses = self.output_responses[self.output_key_to_idx[k]].data
                v.activation = "_".join(
                    self.output_activations[self.output_key_to_idx[k]].__name__.split(
                        "_"
                    )[:-1]
                )
            else:
                v.bias = self.hidden_biases[self.hidden_key_to_idx[k]].data
                v.responses = self.hidden_responses[self.hidden_key_to_idx[k]].data
                v.activation = "_".join(
                    self.hidden_activations[self.hidden_key_to_idx[k]].__name__.split(
                        "_"
                    )[:-1]
                )
                # v.aggregation = 'sum'
        for k, v in genome.connections.items():
            if v.enabled:
                from_k, to_k = k
                if to_k in self.output_key_to_idx.keys():
                    if from_k in self.input_key_to_idx.keys():
                        v.weight = self.input_to_output[
                            self.input_key_to_idx[from_k], self.output_key_to_idx[to_k]
                        ].data
                    elif from_k in self.hidden_key_to_idx.keys():
                        v.weight = self.hidden_to_output[
                            self.hidden_key_to_idx[from_k], self.output_key_to_idx[to_k]
                        ].data
                    elif from_k in self.output_key_to_idx.keys():
                        v.weight = self.output_to_output[
                            self.output_key_to_idx[from_k], self.output_key_to_idx[to_k]
                        ].data
                else:
                    if from_k in self.input_key_to_idx.keys():
                        v.weight = self.input_to_hidden[
                            self.input_key_to_idx[from_k], self.hidden_key_to_idx[to_k]
                        ].data
                    elif from_k in self.hidden_key_to_idx.keys():
                        v.weight = self.hidden_to_hidden[
                            self.hidden_key_to_idx[from_k], self.hidden_key_to_idx[to_k]
                        ].data
                    elif from_k in self.output_key_to_idx.keys():
                        v.weight = self.output_to_hidden[
                            self.output_key_to_idx[from_k], self.hidden_key_to_idx[to_k]
                        ].data

        return genome


def str_to_tuple_key(str_key):
    return (int(str_key.split(",")[0]), int(str_key.split(",")[1]))


def tuple_to_str_key(tuple_key):
    return f"{tuple_key[0]},{tuple_key[1]}"


class RecurrentNetworkAgg(nn.Module):
    def __init__(
        self,
        input_neuron_ids,
        hidden_neuron_ids,
        output_neuron_ids,
        biases,
        responses,
        activations,
        aggregations,
        connections,
        device="cpu",
    ):
        """
        :param input_neuron_ids: list of ids of input neurons
        :param hidden_neuron_ids: list of ids of hidden neurons
        :param output_neuron_ids: list of ids of output neurons
        """
        super().__init__()
        self.device = device

        # input neurons are indexed from -n_inputs
        self.input_neuron_ids = input_neuron_ids
        self.n_inputs = len(input_neuron_ids)
        self.output_neuron_ids = output_neuron_ids  # output neurons are indexed from 0
        self.n_outputs = len(output_neuron_ids)
        # hidden neurons are indexed from n_outputs
        self.hidden_neuron_ids = hidden_neuron_ids
        self.n_hidden = len(hidden_neuron_ids)

        # connection parameters
        # torch accept only str as parameter name and not Tuple of int
        connections_with_str_keys = dict()
        for k, v in connections.items():
            connections_with_str_keys[tuple_to_str_key(k)] = nn.Parameter(
                torch.tensor(v)
            )
        self.connections = nn.ParameterDict(connections_with_str_keys)

        # node parameters : just list where ids are node keys
        biases_with_str_keys = dict()
        for k, v in biases.items():
            biases_with_str_keys[str(k)] = nn.Parameter(torch.tensor(v))
        self.biases = nn.ParameterDict(biases_with_str_keys)
        responses_with_str_keys = dict()
        for k, v in responses.items():
            responses_with_str_keys[str(k)] = nn.Parameter(torch.tensor(v))
        self.responses = nn.ParameterDict(responses_with_str_keys)
        self.activation_functions = dict()
        for k, v in activations.items():
            self.activation_functions[str(k)] = v
        self.aggregation_functions = dict()
        for k, v in aggregations.items():
            self.aggregation_functions[str(k)] = v

        self.input_key2idx = dict.fromkeys(self.input_neuron_ids)
        v = 0
        for k in self.input_key2idx.keys():
            self.input_key2idx[k] = v
            v += 1

        self.neuron_key2idx = dict.fromkeys(
            self.output_neuron_ids + self.hidden_neuron_ids
        )
        v = 0
        for k in self.neuron_key2idx.keys():
            self.neuron_key2idx[k] = v
            v += 1

        # push to device
        self.to(self.device)

    def forward(self, inputs):
        """
        :param inputs: tensor of size (batch_size, n_inputs)

        Note: output of a node as follows: activation(bias+(response∗aggregation(inputs)))

        returns: (batch_size, n_outputs)
        """
        batch_size = len(inputs)
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        after_pass_node_activs = torch.zeros_like(self.node_activs).to(self.device)

        for neuron_key in self.output_neuron_ids + self.hidden_neuron_ids:
            incoming_connections = [
                c
                for c in self.connections.keys()
                if str_to_tuple_key(c)[1] == neuron_key
            ]
            incoming_inputs = torch.empty((0, batch_size)).to(self.device)

            # multiply by connection weights
            for conn in incoming_connections:
                input_key = str_to_tuple_key(conn)[0]
                # input_to_neuron case
                if input_key < 0:
                    incoming_inputs = torch.vstack(
                        [
                            incoming_inputs,
                            self.connections[conn]
                            * inputs[:, self.input_key2idx[input_key]],
                        ]
                    )
                # neuron_to_neuron case
                else:
                    incoming_inputs = torch.vstack(
                        [
                            incoming_inputs,
                            self.connections[conn]
                            * self.node_activs[:, self.neuron_key2idx[input_key]],
                        ]
                    )

            # aggregate incoming inputs
            if len(incoming_connections) > 0:
                node_outputs = self.activation_functions[str(neuron_key)](
                    self.biases[str(neuron_key)]
                    + self.responses[str(neuron_key)]
                    * self.aggregation_functions[str(neuron_key)](incoming_inputs.t())
                )
            else:
                node_outputs = torch.zeros((batch_size))
            after_pass_node_activs[:, self.neuron_key2idx[neuron_key]] = node_outputs

        self.node_activs = after_pass_node_activs
        return after_pass_node_activs[:, self.output_neuron_ids]

    def activate(self, inputs, n_passes=1):
        """
        :param inputs: tensor of size (..., n_inputs) => only the last dim matter, then CPPN is applied on all coordinates
        :param n_passes: number of passes in the RNN TODO: currently global passes should it be internal passes as in pytorch-neat by uber research ?
        returns: (..., n_outputs)
        """
        input_size = inputs.shape[:-1]
        batch_size = torch.prod(torch.tensor(input_size))
        assert inputs.shape[-1] == self.n_inputs
        inputs = inputs.view(-1, self.n_inputs)

        self.node_activs = torch.zeros((batch_size, self.n_outputs + self.n_hidden)).to(
            self.device
        )
        for _ in range(n_passes):
            outputs = self.forward(inputs)

        outputs = outputs.view(input_size + (self.n_outputs,))
        return outputs

    def draw_net(
        self,
        config=None,
        view=False,
        filename=None,
        node_names=None,
        node_colors=None,
        fmt="svg",
    ):
        if node_names is None:
            node_names = {}

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            "shape": "circle",
            "fontsize": "9",
            "height": "0.2",
            "width": "0.2",
        }

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        for k in self.input_neuron_ids:
            name = node_names.get(k, str(k))
            input_attrs = {
                "style": "filled",
                "shape": "box",
                "fillcolor": node_colors.get(k, "lightgray"),
            }
            dot.node(name, _attributes=input_attrs)

        for k in self.output_neuron_ids:
            name = node_names.get(k, str(k))
            node_attrs = {
                "style": "filled",
                "shape": "box",
                "fillcolor": node_colors.get(k, "lightblue"),
                "fontsize": "9",
                "fontcolor": node_colors.get(k, "blue"),
                "xlabel": f"{self.activation_functions[str(k)].__name__[:-11]}({self.biases[str(k)]:.1f}+\\n{self.responses[str(k)]:.1f}*{self.aggregation_functions[str(k)].__name__[:-12]}(inputs))",
            }

            dot.node(name, _attributes=node_attrs)

        for n in self.hidden_neuron_ids:
            attrs = {
                "style": "filled",
                "fillcolor": node_colors.get(n, "white"),
                "fontsize": "9",
                "xlabel": f"{self.activation_functions[str(n)].__name__[:-11]}({self.biases[str(n)]:.1f}+\\n{self.responses[str(n)]:.1f}*{self.aggregation_functions[str(n)].__name__[:-12]}(inputs))",
            }
            dot.node(str(n), _attributes=attrs)

        for cg, cg_weight in self.connections.items():
            input = str_to_tuple_key(cg)[0]
            output = str_to_tuple_key(cg)[1]
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid"
            color = "green" if cg_weight > 0 else "red"
            width = str(abs(cg_weight))
            dot.edge(
                a,
                b,
                _attributes={
                    "style": style,
                    "color": color,
                    "fontcolor": color,
                    "penwidth": width,
                    "fontsize": "9",
                    "label": f"w={cg_weight:.1f}",
                },
            )

        dot.render(filename, view=view)

        return dot

    @staticmethod
    def create(genome, config, device="cpu"):
        """Receives a genome and returns the RecurrentCPPN()"""
        genome_config = config.genome_config

        input_neuron_ids = genome_config.input_keys  # list of ids of input neurons
        output_neuron_ids = genome_config.output_keys  # list of ids of output neurons
        hidden_neuron_ids = list(genome.nodes.keys())[
            len(output_neuron_ids) :
        ]  # list of ids of hidden neurons (neither input nor ouput)

        # biases = {node_id: nodes[node_id].bias}
        biases = dict.fromkeys(genome.nodes.keys())
        # responses = {node_id: nodes[node_id].response}
        responses = dict.fromkeys(genome.nodes.keys())
        # activations = {node_id: nodes[node_id].activation}
        activations = dict.fromkeys(genome.nodes.keys())
        # aggregations = {node_id: nodes[node_id].aggregation}
        aggregations = dict.fromkeys(genome.nodes.keys())

        for node_id, node in genome.nodes.items():
            biases[node_id] = node.bias
            responses[node_id] = node.response
            activations[node_id] = str_to_activation[node.activation]
            aggregations[node_id] = str_to_aggregation[node.aggregation]

        connections = dict.fromkeys(
            genome.connections.keys()
        )  # connections = {(from_node_id, to_node_id): connections[(from_node_id, to_node_id)].weight}
        for (from_id, to_id), connection in genome.connections.items():
            if connection.enabled:
                connections[(from_id, to_id)] = connection.weight
            else:
                del connections[(from_id, to_id)]

        return RecurrentNetworkAgg(
            input_neuron_ids,
            hidden_neuron_ids,
            output_neuron_ids,
            biases,
            responses,
            activations,
            aggregations,
            connections,
            device=device,
        )

    def update_genome(self, genome):
        for k, v in genome.nodes.items():
            v.bias = self.biases[str(k)].data
            v.responses = self.responses[str(k)].data
            v.activation = "_".join(
                self.activation_functions[str(k)].__name__.split("_")[:-1]
            )
            v.aggregation = "_".join(
                self.aggregation_functions[str(k)].__name__.split("_")[:-1]
            )
        for k, v in genome.connections.items():
            if v.enabled:
                v.weight = self.connections[tuple_to_str_key(k)].data
