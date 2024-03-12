import torch
from auto_disc.auto_disc.wrappers.BoxProjector import BoxProjector


def test__update_low_high():
    dim = 10
    input = torch.rand(dim)

    box = BoxProjector(premap_key="output")
    box._update_low_high(input)

    assert torch.allclose(box.low, torch.zeros_like(input))
    assert torch.allclose(box.high, input)
    old_low = box.low.clone()
    old_high = box.high.clone()

    new_input = torch.rand(dim) * 10 - 5
    box._update_low_high(new_input)
    upper_mask = torch.greater(new_input, old_high)
    lower_mask = torch.less(new_input, old_low)

    assert torch.allclose(box.low[lower_mask], new_input[lower_mask])
    assert torch.allclose(box.high[upper_mask], new_input[upper_mask])

    # test immutability
    new_input += 2
    assert box.low is not new_input
    assert not torch.allclose(box.low[lower_mask], new_input[lower_mask])


def test__update_low_high_type_conversion():
    dim = 3
    # input is a float tensor
    input = torch.rand(dim)

    # default init_low and init_high are int tensors
    box = BoxProjector(
        premap_key="output",
        init_low=torch.tensor([0, 0, 0]),
        init_high=torch.tensor([0, 0, 0]),
    )
    box._update_low_high(input)

    # low and high should be converted to float for asserts to pass
    assert torch.allclose(box.low, torch.zeros_like(input))
    assert torch.allclose(box.high, input)


def test__clamp_and_truncate():
    dim = 10
    input = torch.rand(dim) + 20

    box = BoxProjector(premap_key="output", bound_upper=torch.tensor([5.0]))
    clamped_output = box._clamp_and_truncate(input)
    assert torch.all(torch.isclose(clamped_output, torch.tensor([5.0])))

    # clamp to a tensor
    box = BoxProjector(
        premap_key="output",
        bound_upper=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )
    clamped_output = box._clamp_and_truncate(input)
    assert torch.all(
        torch.isclose(
            clamped_output,
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        )
    )


def test_sample():
    """
    NOTE: this test is non-deterministic
    """
    dim = 10
    rand_nums_low = torch.rand(dim) * 2 - 4
    rand_nums_high = torch.rand(dim) * 2 + 4

    box = BoxProjector(premap_key="output")
    box.low = rand_nums_low.clone()
    box.high = rand_nums_high.clone()
    box.tensor_shape = rand_nums_low.size()

    for _ in range(100):
        old_sample = box.sample()
        sample = box.sample()

        # samples differ
        assert not torch.all(torch.isclose(sample, old_sample))

        # samples are within the same boundaries
        assert torch.all(torch.greater(sample, rand_nums_low))
        assert torch.all(torch.less(sample, rand_nums_high))


def test_map():
    dim = 10
    input = torch.rand(dim)
    input_dict = {"output": input, "metadata": 1}

    box = BoxProjector(premap_key="output")
    output_dict = box.map(input_dict)

    assert output_dict != input
    assert torch.allclose(output_dict["output"], input_dict["output"])
    assert torch.allclose(box.low, torch.zeros_like(input))
    assert torch.allclose(box.high, input)
    # assert output_dict["sampler"]
    assert output_dict["metadata"] == 1

    old_low = box.low.clone()
    old_high = box.high.clone()

    new_input = torch.rand(dim) * 10 - 5
    input_dict["output"] = new_input
    output_dict = box.map(input_dict)
    upper_mask = torch.greater(new_input, old_high)
    lower_mask = torch.less(new_input, old_low)

    assert torch.allclose(box.low[lower_mask], new_input[lower_mask])
    assert torch.allclose(box.high[upper_mask], new_input[upper_mask])

    assert torch.all(torch.greater(box.sample(), box.low))
    assert torch.all(torch.less(box.sample(), box.high))


def test_map_clamped():
    dim = 10
    input = torch.rand(dim) + 10
    input_dict = {"output": input, "metadata": 1}

    box = BoxProjector(premap_key="output", bound_upper=torch.tensor([1.0]))
    output_dict = box.map(input_dict)

    # will clamp all of these to 1
    assert torch.allclose(output_dict["output"], torch.tensor([1.0]))

    dim = 10
    input = torch.rand(dim) - 10
    input_dict = {"output": input, "metadata": 1}

    box = BoxProjector(premap_key="output", bound_lower=torch.tensor([-1.0]))
    output_dict = box.map(input_dict)

    # will clamp all of these to -1
    assert torch.allclose(output_dict["output"], torch.tensor([-1.0]))
