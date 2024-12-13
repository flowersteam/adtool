import json
from enum import Enum
from typing import Union,  Dict
import numpy as np

import random
import soundfile as sf

class WaveformType(Enum):
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"

class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"

class Generator:
    def __init__(
        self, 
        waveform: WaveformType, 
        frequency: Union[float, 'Generator', 'Filter'], 
        amplitude: Union[float, 'Generator', 'Filter']
    ):
        self.waveform = waveform
        self.frequency = frequency
        self.amplitude = amplitude

    def evaluate(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freq_values = self._resolve(self.frequency, duration, sample_rate)
        amp_values = self._resolve(self.amplitude, duration, sample_rate)

        if self.waveform == WaveformType.SINE:
            signal = np.sin(2 * np.pi * freq_values * time)
        elif self.waveform == WaveformType.SQUARE:
            signal = np.sign(np.sin(2 * np.pi * freq_values * time))
        elif self.waveform == WaveformType.SAWTOOTH:
            signal = 2 * (time * freq_values - np.floor(time * freq_values + 0.5))
        elif self.waveform == WaveformType.TRIANGLE:
            signal = 2 * np.abs(2 * (time * freq_values - np.floor(time * freq_values + 0.5))) - 1
        else:
            signal = np.zeros_like(time)

        return signal * amp_values

    def _resolve(self, value: Union[float, 'Generator', 'Filter'], duration: float, sample_rate: int) -> np.ndarray:
        if isinstance(value, (Generator, Filter)):
            return value.evaluate(duration, sample_rate)
        else:
            return np.full(int(sample_rate * duration), value)

    def to_json(self) -> Dict:
        return {
            "waveform": self.waveform.value,
            "frequency": self._serialize_value(self.frequency),
            "amplitude": self._serialize_value(self.amplitude),
        }

    def _serialize_value(self, value: Union[float, 'Generator', 'Filter']):
        if isinstance(value, (Generator, Filter)):
            return value.to_json()
        else:
            return value

    @staticmethod
    def from_json(data: Dict) -> 'Generator':
        waveform = WaveformType(data["waveform"])
        frequency = Generator._deserialize_value(data["frequency"])
        amplitude = Generator._deserialize_value(data["amplitude"])
        return Generator(waveform, frequency, amplitude)

    @staticmethod
    def _deserialize_value(value):
        if isinstance(value, dict):
            if "filter_type" in value:
                return Filter.from_json(value)
            elif "waveform" in value:
                return Generator.from_json(value)
        return value

class Filter:
    def __init__(
        self,
        input: Union[Generator, 'Filter'],
        filter_type: FilterType,
        cutoff_ratio: float,
    ):
        self.input = input
        self.filter_type = filter_type
        self.cutoff_ratio = cutoff_ratio

        self.filter_order = 2  # Fixed filter order

    def evaluate(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        
        # Evaluate the input signal
        input_signal = self.input.evaluate(duration, sample_rate)
        
        # Get frequency range
        freqs = np.fft.fftfreq(len(input_signal), 1 / sample_rate)
        freqs = np.fft.fftshift(freqs)  # Shift zero frequency to the center
        
        # Cutoff frequency
        cutoff_freq = self.cutoff_ratio * (sample_rate / 2)  # Scale by Nyquist frequency
        
        # Compute the filter
        if self.filter_type == FilterType.LOWPASS:
            filter = 1 / (1 + (freqs / cutoff_freq)**(2 * self.filter_order))
        elif self.filter_type == FilterType.HIGHPASS:
            filter = 1 / (1 + (cutoff_freq / freqs)**(2 * self.filter_order))
        else:
            filter = np.ones_like(freqs)  # Bypass filter for unsupported filter types
        
        # Handle NaN at zero frequency for high-pass
        filter[np.isnan(filter)] = 0
        
        # Apply the filter in the frequency domain
        fft_signal = np.fft.fft(input_signal)
        fft_signal = np.fft.fftshift(fft_signal)  # Align with shifted frequencies
        filtered_fft = fft_signal * filter
        filtered_fft = np.fft.ifftshift(filtered_fft)  # Shift back
        
        # Return the filtered signal
        filtered_signal = np.fft.ifft(filtered_fft).real
        return filtered_signal







    def to_json(self) -> Dict:
        return {
            "input": self.input.to_json(),
            "filter_type": self.filter_type.value,
            "cutoff_ratio": self.cutoff_ratio
        }

    @staticmethod
    def from_json(data: Dict) -> 'Filter':
        input = Generator._deserialize_value(data["input"])
        filter_type = FilterType(data["filter_type"])
        cutoff_ratio = data["cutoff_ratio"]
        return Filter(input, filter_type, cutoff_ratio)




import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from PIL import Image
import base64
import random

class Synth:
    def __init__(self, output: Union[Generator, Filter]):
        self.output = output

    def generate(self, duration: float = 5.0, sample_rate: int = 44100):
        signal = self.output.evaluate(duration, sample_rate)
        return signal

    def to_json(self) -> Dict:
        """
        Convert the synth to a modular JSON format with modules and connections.
        """
        modules = {}
        connections = []

        def add_module(node):
            node_id = id(node)
            if node_id not in modules:
                if isinstance(node, Generator):
                    modules[node_id] = {
                        "type": "Generator",
                        "waveform": node.waveform.value,
                        "frequency": self._serialize_reference(node.frequency),
                        "amplitude": self._serialize_reference(node.amplitude)
                    }
                elif isinstance(node, Filter):
                    modules[node_id] = {
                        "type": "Filter",
                        "filter_type": node.filter_type.value,
                        "cutoff_ratio": node.cutoff_ratio,
                        "input": self._serialize_reference(node.input)
                    }

                # Handle connections
                if isinstance(node, Generator):
                    if isinstance(node.frequency, (Generator, Filter)):
                        connections.append({"from": id(node.frequency), "to": node_id, "type": "frequency"})
                    if isinstance(node.amplitude, (Generator, Filter)):
                        connections.append({"from": id(node.amplitude), "to": node_id, "type": "amplitude"})
                elif isinstance(node, Filter):
                    connections.append({"from": id(node.input), "to": node_id, "type": "input"})

        def traverse(node):
            if isinstance(node, Generator):
                add_module(node)
                if isinstance(node.frequency, (Generator, Filter)):
                    traverse(node.frequency)
                if isinstance(node.amplitude, (Generator, Filter)):
                    traverse(node.amplitude)
            elif isinstance(node, Filter):
                add_module(node)
                if isinstance(node.input, (Generator, Filter)):
                    traverse(node.input)


        traverse(self.output)

        return {
            "modules": modules,
            "connections": connections,
            "output": id(self.output)
        }

    def _serialize_reference(self, value):
        if isinstance(value, (Generator, Filter)):
            return id(value)
        return value

    @staticmethod
    def from_json(data: Dict) -> 'Synth':
        modules = data["modules"]
        module_objects = {}

        def create_module(module_id):
            if module_id in module_objects:
                return module_objects[module_id]
        #    print("modules", modules,"module_id", module_id)
            module_data = modules[module_id]
            if module_data["type"] == "Generator":
                module = Generator(
                    waveform=WaveformType(module_data["waveform"]),
                    frequency=create_module(module_data["frequency"]) if isinstance(module_data["frequency"], int) else module_data["frequency"],
                    amplitude=create_module(module_data["amplitude"]) if isinstance(module_data["amplitude"], int) else module_data["amplitude"]
                )
            elif module_data["type"] == "Filter":
                module = Filter(
                    input=create_module(module_data["input"]),
                    filter_type=FilterType(module_data["filter_type"]),
                    cutoff_ratio=module_data["cutoff_ratio"]
                )
            else:
                raise ValueError("Unknown module type")

            module_objects[module_id] = module
            return module

        output_module = create_module(data["output"])
        return Synth(output=output_module)


    def image(self) -> Image.Image:
        """
        Generates a 512x512 image plotting the acyclic graph of the synth with info.
        """
        graph = nx.DiGraph()

        # Color nodes based on type

        def create_label(node):
            if isinstance(node, Generator):
                label=f"{node.waveform.value}"
                if  isinstance(node.frequency, float):
                    label += f"\nf:{node.frequency:.2f}"
                if  isinstance(node.amplitude, float):
                    label += f"\nAmp:{node.amplitude:.2f}"
                return label
            elif isinstance(node, Filter):
                label=f"{node.filter_type.value}"
                if  isinstance(node.cutoff_ratio, float):
                    label += f"\n{node.cutoff_ratio:.2f}"
                return label
            
            return ""


        def add_node_edges(node, parent_id=None):
            node_id = id(node)
            if isinstance(node, Generator):
                graph.add_node(node_id, label=create_label(node),
                               type="Generator"


                               
                               ) #Generator\nWaveform: 
                if isinstance(node.frequency, (Generator, Filter)):
                    add_node_edges(node.frequency, node_id)
                    graph.add_edge(id(node.frequency), node_id, label="frequency")
                if isinstance(node.amplitude, (Generator, Filter)):
                    add_node_edges(node.amplitude, node_id)
                    graph.add_edge(id(node.amplitude), node_id, label="amplitude")
            elif isinstance(node, Filter):
                graph.add_node(node_id, label=create_label(node),
                                 type="Filter"
                               ) #Filter\nType:  Cutoff: 
                add_node_edges(node.input, node_id)
                graph.add_edge(id(node.input), node_id, label="input")

            if parent_id is not None:
                graph.add_edge(node_id, parent_id)

        add_node_edges(self.output)

        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        plt.figure(figsize=(7, 7))


        # Draw nodes and edges
        labels = nx.get_node_attributes(graph, 'label')
        node_colors = [ "lightblue"
                       if data["type"] == "Generator" 
                       else  "lightgreen"
                       for node_id, data in graph.nodes(data=True)
                       ]

        nx.draw(graph, pos, labels=labels, with_labels=True, node_size=2500, node_color=node_colors, font_size=12, font_color="black")
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

        # Plot waveforms for generators
        for node_id, data in graph.nodes(data=True):
            if "Generator" in data.get("label", ""):
                node = next((n for n in [self.output] if id(n) == node_id), None)
                if node:
                    signal = node.evaluate(duration=0.1, sample_rate=44100)  # Short waveform for visualization
                    time = [i / 44100 for i in range(len(signal))]
                    ax = plt.gca()
                    inset = ax.inset_axes([pos[node_id][0] / 512, pos[node_id][1] / 512, 0.2, 0.2])
                    inset.plot(time, signal, linewidth=2)
                    inset.set_xticks([])
                    inset.set_yticks([])

        # Save the plot to an image
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=512/7)
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close()
        return image



    def add_noise(self):
        """
        Mutates the graph by adding 5% noise to a random numeric value
        of a random node that is not determined by another module.
        """
        # Collect all mutable values
        mutable_nodes = []

        def collect_mutable_nodes(node):
            if isinstance(node, Generator):
                if isinstance(node.frequency, (float, int)):
                    mutable_nodes.append((node, 'frequency'))
                # if isinstance(node.amplitude, (float, int)):
                #     mutable_nodes.append((node, 'amplitude'))
            elif isinstance(node, Filter):
                if isinstance(node.cutoff_ratio, (float, int)):
                    mutable_nodes.append((node, 'cutoff_ratio'))

            if isinstance(node, Generator):
                if isinstance(node.frequency, (Generator, Filter)):
                    collect_mutable_nodes(node.frequency)
                if isinstance(node.amplitude, (Generator, Filter)):
                    collect_mutable_nodes(node.amplitude)
            elif isinstance(node, Filter):
                if isinstance(node.input, (Generator, Filter)):
                    collect_mutable_nodes(node.input)

        collect_mutable_nodes(self.output)

        # Randomly select a node and field
        if not mutable_nodes:
            print("No mutable nodes found!")
            return

        node, field = random.choice(mutable_nodes)

        # Apply 5% noise
        original_value = getattr(node, field)
        noise = original_value * 0.05 * random.choice([-1, 1])
        new_value = original_value + noise
        if isinstance(node, Filter):
            new_value = max(0.1, min(0.9, new_value))

        setattr(node, field, new_value)

     #   print(f"Mutated {field} of {type(node).__name__} (ID {id(node)}) "f"from {original_value} to {getattr(node, field)}")


    def mutate(self):
        if random.random() < 0.9:
            self.add_noise()
        else:
            self.mutate_graph()           




def mutate_graph(self):
    """
    Randomly mutate the acyclic graph by:
    - Adding a filter after any filter or generator
    - Adding a generator modulating a currently float fixed attribute
    - Removing a filter and reconnecting its input to its output
    - Removing a generator and cleaning up its dependents
    """
    nodes = []

    # Traverse the graph to collect all nodes
    def collect_nodes(node):
        nodes.append(node)
        if isinstance(node, Generator):
            if isinstance(node.frequency, (Generator, Filter)):
                collect_nodes(node.frequency)
            if isinstance(node.amplitude, (Generator, Filter)):
                collect_nodes(node.amplitude)
        elif isinstance(node, Filter):
            if isinstance(node.input, (Generator, Filter)):
                collect_nodes(node.input)

    collect_nodes(self.output)

    # Randomly pick a mutation type 
    # count number of each type of node
    num_generators = len([n for n in nodes if isinstance(n, Generator)])
    num_filters = len([n for n in nodes if isinstance(n, Filter)])

    mutation_types = [
        "add_filter" if num_filters < num_generators else None,
        "add_generator" if 5 > num_generators > 0 else None,
        "remove_filter" if num_filters > 0 else None,
        "remove_generator" if num_generators > 1 else None
    ]
    mutation_types = [m for m in mutation_types if m is not None]

    mutation_type = random.choice(mutation_types)

    print(f"{mutation_type} Num generators: {num_generators}, Num filters: {num_filters}")



    if mutation_type == "add_filter":
        # Add a filter after a random filter or generator
        target = random.choice([n for n in nodes if isinstance(n, (Generator, Filter))])
        new_filter = Filter(
            input=target,
            filter_type=random.choice(list(FilterType)),
            cutoff_ratio=random.uniform(0, 1)
        )
        # Replace references to `target` with `new_filter`
        if self.output == target:
            self.output = new_filter
        else:
            for node in nodes:
                if isinstance(node, Generator):
                    if node.frequency == target:
                        node.frequency = new_filter
                    if node.amplitude == target:
                        node.amplitude = new_filter
                elif isinstance(node, Filter):
                    if node.input == target:
                        node.input = new_filter

    elif mutation_type == "add_generator":
        # Add a generator modulating a currently float fixed attribute
        target = random.choice([n for n in nodes if isinstance(n, Generator)])
        attr = random.choice(["frequency", "amplitude"])
        if isinstance(getattr(target, attr), float):
            new_generator = Generator(
                waveform=random.choice(list(WaveformType)),
                frequency=random.uniform(1, 20) if random.random() < 0.5 else random.uniform(100, 1000),
                amplitude=1.0
            )
            setattr(target, attr, new_generator)

    elif mutation_type == "remove_filter":
        # Remove a filter and reconnect its input to its output
        target = random.choice([n for n in nodes if isinstance(n, Filter)])
        input_node = target.input
        if self.output == target:
            self.output = input_node
        else:
            for node in nodes:
                if isinstance(node, Generator):
                    if node.frequency == target:
                        node.frequency = input_node
                    if node.amplitude == target:
                        node.amplitude = input_node
                elif isinstance(node, Filter):
                    if node.input == target:
                        node.input = input_node

    elif mutation_type == "remove_generator":
        # list generators that have a generator in it's dependents recursively, not considering filters
        generators_with_generators = []
        for node in nodes:
            if isinstance(node, Generator):
                if isinstance(node.frequency, Generator) or isinstance(node.amplitude, Generator):
                    generators_with_generators.append(node)

        if generators_with_generators:
            target = random.choice(generators_with_generators)
            # Remove a generator and clean up its dependents
            for node in nodes:
                if isinstance(node, Generator):
                    if node.frequency == target:
                        node.frequency = target.frequency
                    if node.amplitude == target:
                        node.amplitude = target.amplitude

    else:
        print("No valid mutation type found!")


# Add the mutate method to the Synth class
setattr(Synth, "mutate_graph", mutate_graph)

import time
import matplotlib.pyplot as plt

