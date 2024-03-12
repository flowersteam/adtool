import os
import pathlib
import shutil

import pytest
import pytest_mock
from auto_disc.auto_disc.wrappers.SaveWrapper import BufferStreamer, SaveWrapper
from auto_disc.utils.leaf.locators.LinearBase import Stepper
from auto_disc.utils.leaf.locators.locators import LinearLocator


def setup_function(function):
    global RESOURCE_URI
    FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(FILE_PATH, "tmp")
    os.mkdir(RESOURCE_URI)


def teardown_function(function):
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)


def generate_data():
    # creates two checkpoints, the first with a buffer of length 2,
    # the second with a buffer of length 3
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(premap_keys=["a", "b"], postmap_keys=["b", "a"])
    output = input

    for i in range(2):
        output = wrapper.map(output)
        output = wrapper.map(output)

        # vary save buffer length
        if i == 1:
            output = wrapper.map(output)

        leaf_uid = wrapper.save_leaf(RESOURCE_URI)
    return leaf_uid, wrapper


def generate_data_alot():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        premap_keys=["a", "b", "step"], postmap_keys=["b", "a", "step"]
    )
    output = input
    output["step"] = 0

    for _ in range(10):
        output["step"] += 1
        output = wrapper.map(output)
        output = wrapper.map(output)

        leaf_uid = wrapper.save_leaf(RESOURCE_URI)
    return leaf_uid, wrapper


def test_SaveWrapper____init__():
    input = {"in": 1}
    wrapper = SaveWrapper(
        premap_keys=["in"], postmap_keys=["out"], inputs_to_save=["in"]
    )
    wrapper_def = SaveWrapper(premap_keys=["in"], postmap_keys=["out"])
    assert isinstance(wrapper.locator, LinearLocator)
    assert isinstance(wrapper_def.locator, LinearLocator)
    assert wrapper.locator.resource_uri == ""
    assert wrapper_def.locator.resource_uri == ""
    del wrapper.locator
    del wrapper_def.locator
    assert wrapper.__dict__ == wrapper_def.__dict__


def test_SaveWrapper__map():
    input = {"in": 1}
    wrapper = SaveWrapper(premap_keys=["in"], postmap_keys=["out"])
    output = wrapper.map(input)
    assert output["out"] == 1
    assert len(output) == 1
    assert wrapper.buffer == [{"in": 1}]


def test_SaveWrapper__map_default():
    input = {"data": 1}
    wrapper = SaveWrapper()
    output = wrapper.map(input)
    assert output["data"] == 1
    assert len(output) == 1
    assert wrapper.buffer == [{"data": 1}]


def test_SaveWrapper__map_minimal():
    input = {"data": 1, "metadata": 0}
    wrapper = SaveWrapper(premap_keys=["data"], postmap_keys=["data"])
    output = wrapper.map(input)
    assert output["data"] == 1
    assert len(output) == 2
    assert wrapper.buffer == [{"data": 1}]


def test_SaveWrapper__map_complex():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(premap_keys=["a", "b"], postmap_keys=["b", "a"])
    output = wrapper.map(input)
    assert output["a"] == 2
    assert output["b"] == 1
    assert wrapper.buffer == [{"a": 1, "b": 2}]

    wrapper.map(output)
    assert wrapper.buffer == [{"a": 1, "b": 2}, {"a": 2, "b": 1}]


def test_SaveWrapper__serialize():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(premap_keys=["a", "b"], postmap_keys=["b", "a"])
    output = wrapper.map(input)
    wrapper.map(output)
    bin = wrapper.serialize()

    linear = LinearLocator()
    _, data_bin = LinearLocator.parse_bin(bin)
    a = Stepper().deserialize(data_bin)
    assert a.buffer == wrapper.buffer


def test_SaveWrapper__saveload_basic():
    """
    This tests saving and loading of a single save step (which saves two
    "map" steps of progress)
    """
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(premap_keys=["a", "b"], postmap_keys=["b", "a"])

    output = wrapper.map(input)
    wrapper.map(output)

    leaf_uid = wrapper.save_leaf(RESOURCE_URI)

    # retrieve from leaf nodes of tree
    new_wrapper = SaveWrapper()
    wrapper_loaded = new_wrapper.load_leaf(leaf_uid, RESOURCE_URI)
    buffer = wrapper_loaded.buffer

    # unpack and check loaded Stepper
    assert len(buffer) == 2
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}


def test_SaveWrapper__saveload_advanced():
    """
    This tests saving and loading of multiple save steps (which saves two
    "map" steps of progress)
    """
    leaf_uid, _ = generate_data()

    # retrieve from leaf nodes of tree
    new_wrapper = SaveWrapper()
    wrapper_loaded = new_wrapper.load_leaf(leaf_uid, RESOURCE_URI)
    buffer = wrapper_loaded.buffer

    # unpack and check loaded Stepper
    assert len(buffer) == 3
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}

    # check metadata
    assert wrapper_loaded.inputs_to_save == ["a", "b"]


def test_SaveWrapper__saveload_whole_history():
    leaf_uid, _ = generate_data()

    def retrieve_func(length):
        new_wrapper = SaveWrapper()
        wrapper_loaded = new_wrapper.load_leaf(leaf_uid, RESOURCE_URI, length=length)
        buffer = wrapper_loaded.buffer
        return buffer

    # try retrieval of entire sequence
    buffer = retrieve_func(length=-1)

    assert len(buffer) == 5
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 2, "b": 1}
    assert buffer[4] == {"a": 1, "b": 2}

    # try retrieval of entire sequence with explicit length
    # note that the length argument does not correspond
    # to the length of the buffer, because of save frequency
    buffer = retrieve_func(length=2)

    assert len(buffer) == 5
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 2, "b": 1}
    assert buffer[4] == {"a": 1, "b": 2}

    buffer = retrieve_func(length=1)

    assert len(buffer) == 3
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}


def test_SaveWrapper_get_history():
    leaf_uid, wrapper = generate_data()
    # remember old parent_id to check it will not change
    old_parent_id = wrapper.locator.parent_id
    # load something in the buffer that will be unchanged
    test_input = {"a": 100, "b": 200}
    wrapper.map(test_input)
    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id

    # try retrieval of entire sequence
    buffer = wrapper.get_history(lookback_length=-1)

    assert len(buffer) == 6
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 2, "b": 1}
    assert buffer[4] == {"a": 1, "b": 2}
    assert buffer[5] == {"a": 100, "b": 200}

    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id

    # try retrieval of partial sequence
    buffer = wrapper.get_history(lookback_length=2)

    assert len(buffer) == 4
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 100, "b": 200}

    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id

    # try retrieval of singleton
    buffer = wrapper.get_history()

    assert len(buffer) == 1
    assert buffer[0] == {"a": 100, "b": 200}
    # check that the buffer is not the same object in memory,
    # i.e., it was deepcopied
    assert buffer is not wrapper.buffer

    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id

    # check trivial case
    with pytest.raises(ValueError):
        buffer = wrapper.get_history(lookback_length=0)


def test_SaveWrapper__retrieve_buffer():
    leaf_uid, wrapper = generate_data()
    # remember old parent_id to check it will not change
    old_parent_id = wrapper.locator.parent_id
    # load something in the buffer that will be unchanged
    test_input = {"a": 100, "b": 200}
    wrapper.map(test_input)
    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id

    # try retrieval of entire sequence
    buffer = wrapper._retrieve_buffer(RESOURCE_URI, length=-1)

    assert len(buffer) == 5
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 2, "b": 1}
    assert buffer[4] == {"a": 1, "b": 2}

    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id

    # try retrieval of entire sequence with explicit length
    buffer = wrapper._retrieve_buffer(RESOURCE_URI, length=2)

    assert len(buffer) == 5
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 2, "b": 1}
    assert buffer[4] == {"a": 1, "b": 2}

    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id

    # try retrieval of entire sequence with short length
    buffer = wrapper._retrieve_buffer(RESOURCE_URI, length=1)

    assert len(buffer) == 3
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}

    assert wrapper.buffer == [{"a": 100, "b": 200}]
    assert wrapper.locator.parent_id == old_parent_id


def test_SaveWrapper_generate_dataloader(mocker):
    leaf_uid, wrapper = generate_data_alot()
    # remember old parent_id to check it will not change
    old_parent_id = wrapper.locator.parent_id
    # load something in the buffer that will be unchanged
    test_input = {"a": 100, "b": 200, "step": -1}
    wrapper.map(test_input)
    assert wrapper.buffer == [{"a": 100, "b": 200, "step": -1}]
    assert wrapper.locator.parent_id == old_parent_id

    # use pytest to spy on the _next_cachebuf method
    spy = mocker.spy(BufferStreamer, "_next_cachebuf")

    # try retrieval of entire sequence
    dataloader = wrapper.generate_dataloader(RESOURCE_URI, cachebuf_size=2)
    full_history = []
    for i, el in enumerate(dataloader):
        full_history.append(el)
    assert len(full_history) == 20
    assert full_history[0]["step"] == 10
    assert full_history[1]["step"] == 10
    assert full_history[-2]["step"] == 1
    assert full_history[-1]["step"] == 1

    assert wrapper.buffer == [{"a": 100, "b": 200, "step": -1}]
    assert wrapper.locator.parent_id == old_parent_id

    assert spy.call_count == 5


def test_BufferStreamer___init__():
    leaf_uid, wrapper = generate_data_alot()
    streamer = BufferStreamer(wrapper, resource_uri=RESOURCE_URI, cachebuf_size=2)
    assert streamer.cachebuf_size == 2
    assert streamer.db_url
    assert streamer._i == 10


def test_BufferStreamer__get_db_name():
    leaf_uid, wrapper = generate_data_alot()
    streamer = BufferStreamer(wrapper, resource_uri=RESOURCE_URI, cachebuf_size=2)

    # saving done in generate_data_alot will create a single folder to check
    file_list = os.listdir(RESOURCE_URI)
    retrieved_uri = file_list[0]

    assert streamer._get_db_name() == retrieved_uri


def test_BufferStreamer__next_cachebuf():
    leaf_uid, wrapper = generate_data_alot()
    streamer = BufferStreamer(wrapper, resource_uri=RESOURCE_URI, cachebuf_size=2)

    # check that the first cachebuf is correct
    cachebuf = streamer._next_cachebuf()
    assert len(cachebuf) == 4
    assert cachebuf == [
        {"a": 1, "b": 2, "step": 9},
        {"a": 2, "b": 1, "step": 9},
        {"a": 1, "b": 2, "step": 10},
        {"a": 2, "b": 1, "step": 10},
    ]
    assert streamer._i == 8

    # check that the second cachebuf is correct
    cachebuf = streamer._next_cachebuf()
    assert len(cachebuf) == 4
    assert cachebuf == [
        {"a": 1, "b": 2, "step": 7},
        {"a": 2, "b": 1, "step": 7},
        {"a": 1, "b": 2, "step": 8},
        {"a": 2, "b": 1, "step": 8},
    ]
    assert streamer._i == 6

    # catch the guard
    with pytest.raises(Exception) as excinfo:
        streamer._i = 0
        streamer._next_cachebuf()
        assert excinfo.value == "This should be unreachable."


def test_BufferStreamer___next__():
    leaf_uid, wrapper = generate_data_alot()
    streamer = BufferStreamer(wrapper, resource_uri=RESOURCE_URI, cachebuf_size=2)
    assert streamer.__next__() == {"a": 2, "b": 1, "step": 10}
    assert streamer.__next__() == {"a": 1, "b": 2, "step": 10}
    assert streamer.__next__() == {"a": 2, "b": 1, "step": 9}
    assert streamer.__next__() == {"a": 1, "b": 2, "step": 9}


# TODO: enable batching, see gh issue #38
# def test_BufferStreamer___iter__():
#     leaf_uid, wrapper = generate_data_alot()
#     streamer = BufferStreamer(
#         wrapper, resource_uri=RESOURCE_URI, cachebuf_size=2, mode="batched"
#     )
#     iterable_streamer = streamer.__iter__()
#     streamer = BufferStreamer(
#         wrapper, resource_uri=RESOURCE_URI, cachebuf_size=2, mode="serial"
#     )
#     assert isinstance(iterable_streamer, BufferStreamer)
#     assert iterable_streamer.__next__ == streamer._next_batched
#     assert iterable_streamer.__next__() == [
#         {"a": 2, "b": 1, "step": 10},
#         {"a": 1, "b": 2, "step": 10},
#     ]


def test_BufferStreamer_iterating():
    leaf_uid, wrapper = generate_data_alot()
    streamer = BufferStreamer(wrapper, resource_uri=RESOURCE_URI, cachebuf_size=2)
    step_array = []
    for output_dict in streamer:
        step_array.append(output_dict["step"])

    assert step_array == [10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]


def test_BufferStreamer_iterating_misaligned():
    leaf_uid, wrapper = generate_data_alot()
    # cachebuf size is not a divisor of the total tree depth
    streamer = BufferStreamer(wrapper, resource_uri=RESOURCE_URI, cachebuf_size=3)
    step_array = []
    for output_dict in streamer:
        step_array.append(output_dict["step"])

    assert step_array == [10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
