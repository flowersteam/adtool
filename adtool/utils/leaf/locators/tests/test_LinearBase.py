import os
import pathlib
import pickle
import shutil
from hashlib import sha1

import adtool.utils.leaf.locators.LinearBase as LinearBase
import pytest
from adtool.utils.leaf.locators.LinearBase import (
    FileLinearLocator,
    Stepper,
    _EngineContext,
)
from sqlalchemy import text


def setup_function(function):
    import sqlite3

    global FILE_PATH, DB_PATH, SCRIPT_PATH, DB_NAME

    SCRIPT_PATH = str(pathlib.Path(__file__).parent.resolve())
    FILE_PATH = os.path.join(SCRIPT_PATH, "tmp")
    os.mkdir(FILE_PATH)
    DB_NAME = get_db_name()
    os.mkdir(os.path.join(FILE_PATH, DB_NAME))
    DB_REL_PATH = f"/{DB_NAME}/lineardb"
    SCRIPT_REL_PATH = "/mockDB.sql"
    DB_PATH = FILE_PATH + DB_REL_PATH
    SCRIPT_PATH = SCRIPT_PATH + SCRIPT_REL_PATH


def teardown_function(function):
    if os.path.exists(FILE_PATH):
        shutil.rmtree(FILE_PATH)


def generate_mock_binary() -> bytes:
    def _pad_binary(bin: bytes) -> bytes:
        """
        Converts the output of Leaf.serialize() into a padded binary format
        with 20-byte SHA1 hash of Leaf metadata, magic byte sequence, and the
        original binary
        """
        stepper = pickle.loads(bin)
        del stepper.buffer
        sha1_hash = FileLinearLocator.hash(pickle.dumps(stepper))
        output_bin = bytes.fromhex(sha1_hash) + bytes.fromhex("deadbeef") + bin
        return output_bin

    s = Stepper()
    query_trajectory = [bytes(1), bytes(2), bytes(4), bytes(9)]
    s.buffer = query_trajectory
    bin = s.serialize()
    padded_bin = _pad_binary(bin)
    return padded_bin, bin


def generate_fake_data(db_url: str):
    import sqlite3

    con = sqlite3.connect(db_url)
    cur = con.cursor()
    with open(SCRIPT_PATH) as f:
        query_string = f.read()
        cur.executescript(query_string)
    return


def get_db_name() -> str:
    padded_bin, _ = generate_mock_binary()
    db_name, _ = FileLinearLocator.parse_bin(padded_bin)
    return db_name


def test_FileLinearLocator___init__():
    x = FileLinearLocator(FILE_PATH)
    assert x.resource_uri == FILE_PATH


def test_FileLinearLocator__init_db():
    x = FileLinearLocator(FILE_PATH)
    LinearBase.init_db(DB_PATH)
    assert os.path.exists(DB_PATH)


def test_FileLinearLocator__insert_node():
    x = FileLinearLocator(FILE_PATH)
    LinearBase.init_db(DB_PATH)

    def get_trajectory_table_length(engine):
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * from trajectories"))
            length_table = len(result.all())
        return length_table

    def get_newest_insert(engine):
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM trajectories ORDER BY id DESC LIMIT 1")
            )
            new_row = result.one()
        return new_row

    with _EngineContext(DB_PATH) as engine:
        length_table = get_trajectory_table_length(engine)
        LinearBase.insert_node(engine, 1)
        new_length_table = get_trajectory_table_length(engine)
        new_row = get_newest_insert(engine)

    # assert new row is inserted properly into trajectories table
    assert new_length_table == length_table + 1
    assert new_row[1] == 1


def test_FileLinearLocator__get_trajectory():
    x = FileLinearLocator(FILE_PATH)
    LinearBase.init_db(DB_PATH)
    generate_fake_data(DB_PATH)

    with _EngineContext(DB_PATH) as engine:
        # test retrieve all
        _, trajectory, depths = LinearBase._get_trajectory_raw(engine, 5, -1)
        assert trajectory == [bytes(1), bytes(2), bytes(3), bytes(4), bytes(5)]
        assert len(trajectory) - 1 == depths[0]
        # test retrieve one
        _, trajectory, depths = LinearBase._get_trajectory_raw(engine, 5, 1)
        assert trajectory == [bytes(5)]
        assert len(trajectory) - 1 == depths[0]
        # test retrieve arbitrary length with arbitrary id
        _, trajectory, depths = LinearBase._get_trajectory_raw(engine, 7, 3)
        assert trajectory == [bytes(2), bytes(4), bytes(8)]
        assert len(trajectory) - 1 == depths[0]
        # test retrieving longer trajectory than exists
        # should limit to what is available
        _, trajectory, depths = LinearBase._get_trajectory_raw(engine, 5, 100)
        assert trajectory == [bytes(1), bytes(2), bytes(3), bytes(4), bytes(5)]
        assert len(trajectory) - 1 == depths[0]


def test_FileLinearLocator__parse_bin():
    bin, _ = generate_mock_binary()
    sha1_hash, data_bin = FileLinearLocator.parse_bin(bin)

    stepper = pickle.loads(data_bin)
    assert stepper.buffer == [bytes(1), bytes(2), bytes(4), bytes(9)]

    del stepper.buffer
    assert sha1(pickle.dumps(stepper)).hexdigest() == sha1_hash


def test_FileLinearLocator__parse_leaf_uid():
    test_str = "asdiufgapsudf:2"
    db_name, node_id = FileLinearLocator.parse_leaf_uid(test_str)
    assert db_name == "asdiufgapsudf", 2


def test_FileLinearLocator_store():
    x = FileLinearLocator(FILE_PATH)
    LinearBase.init_db(DB_PATH)
    generate_fake_data(DB_PATH)

    padded_bin, data_bin = generate_mock_binary()

    retrieval_key = x.store(padded_bin, 6)

    # assert that retrieval_key is stored
    # and can successfully retrieve trajectory
    db_name, row_id = FileLinearLocator.parse_leaf_uid(retrieval_key)
    subdir = os.path.join(FILE_PATH, db_name)
    db_url = os.path.join(subdir, "lineardb")

    with _EngineContext(db_url) as engine:
        ids, trajectory, _ = LinearBase._get_trajectory_raw(engine, row_id, -1)
        assert ids == [1, 2, 6, 8]
        assert trajectory == [bytes(1), bytes(2), bytes(4), data_bin]
    assert len(os.listdir(FILE_PATH)) == 1
    assert len(os.listdir(subdir)) == 2


def test_FileLinearLocator_retrieve():
    x = FileLinearLocator(FILE_PATH)
    LinearBase.init_db(DB_PATH)
    padded_bin, data_bin = generate_mock_binary()
    retrieval_key = x.store(padded_bin, -1)
    retrieval_key = x.store(padded_bin, 1)
    assert x.parent_id == 2

    # mock storage of sequence
    mock_retrieval_key = DB_NAME + ":2"
    assert retrieval_key == mock_retrieval_key  # SQLite indexed starting at 1

    x = FileLinearLocator(FILE_PATH)
    assert x.parent_id == -1

    bin = x.retrieve(retrieval_key, length=-1)
    assert x.parent_id == 2
    loaded_obj = Stepper().deserialize(bin)

    assert loaded_obj.buffer == [
        bytes(1),
        bytes(2),
        bytes(4),
        bytes(9),
        bytes(1),
        bytes(2),
        bytes(4),
        bytes(9),
    ]


def test_FileLinearLocator_branching():
    x = FileLinearLocator(FILE_PATH)
    LinearBase.init_db(DB_PATH)
    padded_bin, data_bin = generate_mock_binary()
    # store root node
    retrieval_key = x.store(padded_bin, parent_id=-1)
    root_retrieval_key = retrieval_key

    # store depth = 1 node (NOTE: 1-indexing by SQLite means parent here is 1)
    retrieval_key = x.store(padded_bin, parent_id=1)
    second_retrieval_key = retrieval_key

    mock_retrieval_key = DB_NAME + ":1"
    assert root_retrieval_key == mock_retrieval_key  # SQLite indexed starting at 1

    # retrieve original
    x = FileLinearLocator(FILE_PATH)
    bin = x.retrieve(root_retrieval_key, -1)
    loaded_obj = Stepper().deserialize(bin)
    assert loaded_obj.buffer == [bytes(1), bytes(2), bytes(4), bytes(9)]

    # branch
    third_retrieval_key = x.store(padded_bin, parent_id=1)
    assert int(third_retrieval_key.split(":")[-1]) == 3
    assert third_retrieval_key != second_retrieval_key

    x = FileLinearLocator(FILE_PATH)
    bin = x.retrieve(third_retrieval_key, -1)
    loaded_obj = Stepper().deserialize(bin)
    assert loaded_obj.buffer == [
        bytes(1),
        bytes(2),
        bytes(4),
        bytes(9),
        bytes(1),
        bytes(2),
        bytes(4),
        bytes(9),
    ]

    x = FileLinearLocator(FILE_PATH)
    bin = x.retrieve(second_retrieval_key, -1)
    loaded_obj = Stepper().deserialize(bin)
    assert loaded_obj.buffer == [
        bytes(1),
        bytes(2),
        bytes(4),
        bytes(9),
        bytes(1),
        bytes(2),
        bytes(4),
        bytes(9),
    ]
