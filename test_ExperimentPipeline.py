import os
import pathlib
import shutil
import torch


from adtool.maps.MeanBehaviorMap import MeanBehaviorMap
from adtool.maps.UniformParameterMap import UniformParameterMap
from examples.exponential_mixture.systems.ExponentialMixture import ExponentialMixture
from adtool.explorers.IMGEPExplorer import IMGEPExplorer
from adtool.wrappers.IdentityWrapper import IdentityWrapper
from adtool.ExperimentPipeline import ExperimentPipeline
from adtool.callbacks.on_save_callbacks.save_leaf_callback import (
    SaveLeaf,
)
from adtool.utils.logger import AutoDiscLogger


def setup_function(function):
    global RESOURCE_URI
    file_path = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(file_path, "tmp")
    os.mkdir(RESOURCE_URI)
    return


def teardown_function(function):
    global RESOURCE_URI
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)
    return


def callback(**__kwargs):
    print(f"Callback was called on pipeline with" f"kwargs {__kwargs}")
    return


def test_dummy_callback(capsys):
    callback(config=1, metadata=2)
    captured = capsys.readouterr()
    assert captured.out != ""



experiment_id = 1
seed = 1
system_input_key = "params"
system_output_key = "output"

system = ExponentialMixture(sequence_max=10, sequence_density=10)
mean_map = MeanBehaviorMap(premap_key=system_output_key)
param_map = UniformParameterMap(
    premap_key=system_input_key,
    tensor_low=torch.tensor([0.0, 0.0, 0.0]),
    tensor_high=torch.tensor([3.0, 3.0, 3.0]),
)

explorer = IMGEPExplorer(
    premap_key=system_output_key,
    postmap_key=system_input_key,
    parameter_map="adtool.maps.UniformParameterMap.UniformParameterMap",
    behavior_map="adtool.maps.MeanBehaviorMap.MeanBehaviorMap",
    parameter_map_config=
    {
        "tensor_low": torch.tensor([0.0, 0.0, 0.0]),
        "tensor_high": torch.tensor([3.0, 3.0, 3.0]),
    },
    behavior_map_config={
    }
    ,
    equil_time=2,
)(
    system=system
)

pipeline = ExperimentPipeline(
    experiment_id=experiment_id, seed=seed, system=system, explorer=explorer,
    config={},
)

RESOURCE_URI="/tmp"

uid = pipeline.save_leaf(resource_uri=RESOURCE_URI)
print(uid)
x = ExperimentPipeline()
new_pipeline = x.load_leaf(uid, resource_uri=RESOURCE_URI)

print(new_pipeline._explorer)

# check explorer state
assert new_pipeline._explorer.premap_key == system_output_key
assert new_pipeline._explorer.postmap_key == system_input_key
assert torch.allclose(
    new_pipeline._explorer.parameter_map.projector.low,
    torch.tensor([0.0, 0.0, 0.0]),
)
assert torch.allclose(
    new_pipeline._explorer.parameter_map.projector.high,
    torch.tensor([3.0, 3.0, 3.0]),
)
# check system type
assert isinstance(new_pipeline._system, ExponentialMixture)
# check container pointers
assert new_pipeline._explorer._container_ptr == new_pipeline