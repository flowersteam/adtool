import torch
from adtool.wrappers.mutators import add_gaussian_noise


def test_broadcast():
    input_tensor = torch.rand(10)
    output_tensor = add_gaussian_noise(input_tensor, mean=0, std=1)

    assert torch.all(torch.not_equal(input_tensor, output_tensor))

    input_tensor = torch.rand(3)
    output_tensor = add_gaussian_noise(
        input_tensor, mean=torch.tensor([0.0, 0.0, 10000.0]), std=1
    )

    # NOTE: not deterministic, but pretty likely
    assert (output_tensor[2] - input_tensor[2]) > 100
    assert (output_tensor[1] - input_tensor[1]) < 100
    assert (output_tensor[0] - input_tensor[0]) < 100
