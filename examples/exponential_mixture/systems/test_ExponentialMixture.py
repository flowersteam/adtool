import torch
from examples.exponential_mixture.systems.ExponentialMixture import ExponentialMixture


def test_init():
    system = ExponentialMixture(sequence_max=13.0, sequence_density=1313)
    assert system.sequence_max == 13.0
    assert system.sequence_density == 1313


def test_map():
    test_params = torch.rand(100)
    sequence_max = 1000.0
    sequence_density = 500
    input_dict = {"params": test_params}

    system = ExponentialMixture(
        sequence_max=sequence_max, sequence_density=sequence_density
    )
    output_dict = system.map(input_dict)
    assert output_dict["output"].size() == torch.Size([sequence_density])
    assert torch.all(torch.greater(output_dict["output"], 0))


def test_render():
    test_params = torch.rand(100)
    sequence_max = 1000.0
    sequence_density = 500
    input_dict = {"params": test_params}

    system = ExponentialMixture(
        sequence_max=sequence_max, sequence_density=sequence_density
    )
    output_dict = system.map(input_dict)

    byte_img = system.render(output_dict)

    assert isinstance(byte_img, bytes)
