import json
import os
import pathlib
import shutil
import subprocess

import auto_disc.run as run
from auto_disc.ExperimentPipeline import ExperimentPipeline


def setup_function(function):
    global RESOURCE_URI, config_json
    file_path = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(file_path, "tmp")
    neat_config_path = os.path.join(file_path, "neat_config_test.cfg")
    os.mkdir(RESOURCE_URI)
    config_json_exp_mix = {
        "experiment": {
            "name": "newarch_demo",
            "config": {
                "host": "local",
                "save_location": f"{RESOURCE_URI}",
                "nb_seeds": 1,
                "nb_iterations": 20,
                "save_frequency": 1,
            },
        },
        "system": {
            "name": "adtool_default.systems.ExponentialMixture.ExponentialMixture",
            "config": {"sequence_max": 1, "sequence_density": 20},
        },
        "explorer": {
            "name": "auto_disc.auto_disc.explorers.IMGEPFactory",
            "config": {
                "equil_time": 2,
                "param_dim": 1,
                "param_init_low": 0.0,
                "param_init_high": 1.0,
            },
        },
        "callbacks": {},
        "logger_handlers": [],
    }
    config_json_lenia = {
        "experiment": {
            "name": "demo",
            "config": {
                "host": "local",
                "nb_seeds": 1,
                "nb_iterations": 3,
                "save_location": f"{RESOURCE_URI}",
                "save_frequency": 1,
                "discovery_saving_keys": [],
            },
        },
        "system": {
            "name": "adtool_default.systems.LeniaCPPN.LeniaCPPN",
            "config": {
                "SX": 64,
                "SY": 64,
                "version": "pytorch_fft",
                "final_step": 200,
                "scale_init_state": 1,
            },
        },
        "explorer": {
            "name": "auto_disc.auto_disc.explorers.IMGEPFactory",
            "config": {
                "mutator": "specific",
                "equil_time": 1,
                "behavior_map": "LeniaStatistics",
                "parameter_map": "LeniaParameterMap",
                "mutator_config": {},
                "behavior_map_config": {"SX": 64, "SY": 64},
                "parameter_map_config": {
                    "init_state_dim": [64, 64],
                    "neat_config_path": f"{neat_config_path}",
                },
            },
        },
        "input_wrappers": [],
        "output_representations": [],
        "callbacks": {},
        "logger_handlers": [],
    }

    json_requests_dir = os.path.join(file_path, "integration")
    demo_path = os.path.join(json_requests_dir, "demo.json")
    with open(demo_path, "r") as f:
        config_json = json.loads(f.read())

        # override with testing mock
        config_json["experiment"]["config"]["save_location"] = f"{RESOURCE_URI}"
    return


def teardown_function(function):
    global RESOURCE_URI
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)
    return


def test_create():
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)
    assert isinstance(pipeline, ExperimentPipeline)


def test_run():
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)

    run.start(pipeline, 10)


def test_save_SaveDiscoveryOnDisk_cli_entrypoint():
    """Same test as above but tested by calling the script from a shell
    subprocess, mimicking what a user might do."""
    experiment_id = 1
    seed = 1
    config_path = os.path.join(RESOURCE_URI, "config.json")
    with open(config_path, "w") as f:
        config_json["callbacks"] = {"on_discovery": [{"name": "disk", "config": {}}]}
        json.dump(config_json, f)
    proc = subprocess.run(
        [
            "python",
            "run.py",
            "--config_file",
            str(config_path),
            "--experiment_id",
            str(experiment_id),
            "--seed",
            str(seed),
            "--nb_iterations",
            str(10),
        ],
        shell=True,
    )
    assert proc.returncode == 0


def test_save_SaveDiscoveryOnDisk():
    config_json["callbacks"] = {"on_discovery": [{"name": "disk", "config": {}}]}
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)

    run.start(pipeline, 10)

    # rough check of file tree
    files = os.listdir(RESOURCE_URI)
    assert len(files) > 0
    disc_dirs = []

    for f in files:
        tf = f.split("_")
        if (len(tf) > 1) and (tf[-2] == "idx"):
            disc_dirs.append(f)
        else:
            pass

    for dir in disc_dirs:
        each_discovery = os.listdir(os.path.join(RESOURCE_URI, dir))
        assert "rendered_output" in each_discovery
        assert "discovery.json" in each_discovery


def test_save_SaveDiscoveryOnDisk_cli_entrypoint():
    """Same test as above but tested by calling the script from a shell
    subprocess, mimicking what a user might do."""
    experiment_id = 1
    seed = 1
    config_path = os.path.join(RESOURCE_URI, "config.json")
    with open(config_path, "w") as f:
        config_json["callbacks"] = {"on_discovery": [{"name": "disk", "config": {}}]}
        json.dump(config_json, f)
    proc = subprocess.run(
        [
            "python",
            "run.py",
            "--config_file",
            str(config_path),
            "--experiment_id",
            str(experiment_id),
            "--seed",
            str(seed),
            "--nb_iterations",
            str(10),
        ],
        shell=True,
    )
    files = os.listdir(RESOURCE_URI)
    assert len(files) > 0
    disc_dirs = []

    for f in files:
        tf = f.split("_")
        if (len(tf) > 1) and (tf[-2] == "idx"):
            disc_dirs.append(f)
        else:
            pass

    for dir in disc_dirs:
        each_discovery = os.listdir(os.path.join(RESOURCE_URI, dir))
        assert "rendered_output" in each_discovery
        assert "discovery.json" in each_discovery
    assert proc.returncode == 0


def test_save_resume():
    config_json["callbacks"] = {
        "on_saved": [{"name": "base", "config": {}}],
        "on_save_finished": [{"name": "base", "config": {}}],
        "on_discovery": [{"name": "disk", "config": {}}],
    }
    config_json["explorer"]["config"]["equil_time"] = 3
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)
    run.start(pipeline, 3)

    files = os.listdir(RESOURCE_URI)
    reports = []

    for f in files:
        tf = f.split("_")
        if (len(tf) > 1) and (tf[-3] == "idx"):
            reports.append(tf)
        else:
            pass
    # get 2nd experiment
    for r in reports:
        if r[4] == "1":
            uid = r[-1].split(".")[0]
        elif r[4] == "2":
            old_root_uid = r[-1].split(".")[0]
        else:
            pass

    # resume experiment
    config_json["experiment"]["config"]["resume_from_uid"] = uid
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)
    # assert root in the tree is the same
    assert pipeline._explorer._history_saver.locator.parent_id == 2
    # assert run_idx is as saved
    assert pipeline.run_idx == 1
    # assert temporary buffer is restored
    assert len(pipeline._explorer._history_saver.buffer) == 1
    # check callbacks are restored
    assert len(pipeline._on_discovery_callbacks) > 0
    # check resource_uri is restored
    assert pipeline.locator.resource_uri == RESOURCE_URI

    run.start(pipeline, 3)
    files = os.listdir(RESOURCE_URI)
    reports = []

    for f in files:
        tf = f.split("_")
        if (len(tf) > 1) and (tf[-3] == "idx"):
            reports.append(tf)
        else:
            pass

    # check both run_idx 2
    singled = []
    doubled_1 = []
    doubled_2 = []
    for r in reports:
        if r[4] == "0":
            singled.append(r[-1].split(".")[0])
        if r[4] == "1":
            doubled_1.append(r[-1].split(".")[0])
        if r[4] == "2":
            doubled_2.append(r[-1].split(".")[0])
        else:
            pass
    # verifies branches have been created
    assert len(doubled_1) == 2
    assert len(doubled_2) == 2
    assert len(singled) == 1

    # inspect for lineardb filepath
    for root, _, files in os.walk(RESOURCE_URI):
        for f in files:
            if f == "lineardb":
                db_path = os.path.join(root, f)
                break

    # check that the tree is properly updated
    import sqlalchemy

    engine = sqlalchemy.create_engine("sqlite+pysqlite:///" + db_path)
    with engine.connect() as conn:
        output = conn.execute(sqlalchemy.text("SELECT * FROM tree"))

    assert output.fetchall() == [(1, 2), (2, 3), (2, 4), (4, 5)]


def test_save_GenerateReport():
    """
    primarily tests the GenerateReport callback
    """
    config_json["callbacks"] = {"on_save_finished": [{"name": "base", "config": {}}]}
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)
    run.start(pipeline, 1)

    # check file tree
    files = os.listdir(RESOURCE_URI)
    # callback doesn't run if save callback is not provided
    assert len(files) == 0

    # provide the savecallback
    config_json["callbacks"] = {
        "on_save_finished": [{"name": "base", "config": {}}],
        "on_saved": [{"name": "base", "config": {}}],
    }
    experiment_id = 2
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)
    run.start(pipeline, 10)

    # check file tree
    files = os.listdir(RESOURCE_URI)
    # callback runs now
    assert len(files) != 0
    data_dirs = []
    reports = []
    for f in files:
        if f.split(".")[-1] == "json":
            reports.append(f)
        else:
            data_dirs.append(f)

    for r in reports:
        tmp = r.split(".")[0]
        uid = tmp.split("_")[-1]
        assert uid in data_dirs


def test_additional_callback():
    from auto_disc.auto_disc.utils.callbacks.on_discovery_callbacks.save_discovery_on_disk import (
        SaveDiscoveryOnDisk,
    )

    additional_callbacks = {"on_discovery": [SaveDiscoveryOnDisk()]}
    experiment_id = 1
    seed = 1
    pipeline = run.create(
        config_json,
        experiment_id=experiment_id,
        seed=seed,
        additional_callbacks=additional_callbacks,
    )

    run.start(pipeline, 10)

    # rough check of file tree
    files = os.listdir(RESOURCE_URI)
    assert len(files) > 0
    disc_dirs = []

    for f in files:
        tf = f.split("_")
        if (len(tf) > 1) and (tf[-2] == "idx"):
            disc_dirs.append(f)
        else:
            pass

        for dir in disc_dirs:
            each_discovery = os.listdir(os.path.join(RESOURCE_URI, dir))
            assert "rendered_output" in each_discovery
            assert "discovery.json" in each_discovery
