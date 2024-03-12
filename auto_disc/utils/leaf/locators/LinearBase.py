import codecs
import os
import pickle
from hashlib import sha1
from typing import List, Tuple

from auto_disc.utils.leaf.Leaf import Leaf, LeafUID, Locator
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine


def store_metadata(subdir: str, data_bin: bytes) -> None:
    loaded_obj = pickle.loads(data_bin)
    del loaded_obj.buffer
    metadata_bin = pickle.dumps(loaded_obj)
    with open(os.path.join(subdir, "metadata"), "wb") as f:
        f.write(metadata_bin)
    return


def store_data(subdir: str, data_bin: bytes, parent_id: int) -> str:
    """
    Store the unpadded binary data (i.e., without tag).
    """
    db_url = os.path.join(subdir, "lineardb")
    # initializes db if it does not exist
    init_db(db_url)

    with _EngineContext(db_url) as engine:
        # insert node
        delta = convert_bytes_to_base64_str(data_bin)
        row_id = insert_node(engine, delta, parent_id)

    return row_id


def init_db(db_url: str) -> None:
    create_traj_statement = """
    CREATE TABLE trajectories (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        content BLOB NOT NULL
        );
        """
    create_tree_statement = """
    CREATE TABLE tree (
        id INTEGER NOT NULL REFERENCES trajectories(id),
        child_id INTEGER REFERENCES trajectories(id)
        );
        """
    if os.path.exists(db_url):
        return
    else:
        with _EngineContext(db_url) as engine:
            with engine.begin() as con:
                con.execute(text(create_traj_statement))
                con.execute(text(create_tree_statement))
        return


def convert_bytes_to_base64_str(bin: bytes) -> str:
    tmp = codecs.encode(bin, encoding="base64").decode()
    # prune newline
    if tmp[-1] == "\n":
        out_str = tmp[:-1]
    return out_str


def convert_base64_str_to_bytes(b64_str: str) -> bytes:
    return codecs.decode(b64_str.encode(), encoding="base64")


def insert_node(engine: Engine, delta: str, parent_id: int = -1) -> int:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO trajectories (content) VALUES (:z)"), {"z": delta}
        )

        # SQLite exclusive function call to get ID of just inserted row
        res = conn.execute(text("SELECT last_insert_rowid()"))
        id = res.all()[0][0]

        if parent_id != -1:
            conn.execute(
                text("INSERT INTO tree (id, child_id) VALUES (:y, :z)"),
                {"y": parent_id, "z": id},
            )
    return id


def retrieve_trajectory(db_url: str, row_id: int = 1, length: int = 1) -> bytes:
    with _EngineContext(db_url) as engine:
        _, trajectory, _ = _get_trajectory_raw(engine, row_id, length)
        if len(trajectory) == 0:
            raise ValueError("Trajectory has length 0.")
        buffer_concat = []
        for binary in trajectory:
            # this is the same as just Leaf.deserialize(binary)
            # but just in case we want to modify Stepper.deserialize
            # which is the proper interface
            loaded_obj = Stepper().deserialize(binary)
            buffer_concat += loaded_obj.buffer
        loaded_obj.buffer = buffer_concat
        bin = pickle.dumps(loaded_obj)
    return bin


def retrieve_packed_trajectory(
    db_url: str, row_id: int = 1, length: int = 1
) -> Tuple[List[int], List[bytes], List[int]]:
    """
    Simple wrapper which retrieves the raw packed trajectory data
    from db_url.
    """
    with _EngineContext(db_url) as engine:
        ids, packed_traj, depths = _get_trajectory_raw(engine, row_id, length)
    return ids, packed_traj, depths


def _get_trajectory_raw(
    engine, id: int, trajectory_length: int = 1
) -> Tuple[List[int], List[bytes], List[int]]:
    """
    Retrieves trajectory of packed binaries along with DB metadata
    which has HEAD at id and returns the last trajectory_length elements.
    """

    # some ugly casework because I don't know SQLAlchemy
    # but also it's more efficient to do it DB-side and not ORM-side
    query_depth_limited = """
        WITH tree_inheritance AS (
                WITH RECURSIVE cte (id, child_id, depth) AS (
                    SELECT :z, NULL, 0
                    UNION ALL
                    SELECT id, child_id, 1
                        FROM tree WHERE tree.child_id = :z
                    UNION
                    SELECT y.id, y.child_id, depth + 1
                        FROM cte AS x INNER JOIN tree AS y ON y.child_id = x.id
                        -- subquery to limit depth of recursion
                        WHERE (
                            WITH const AS (SELECT :trajectory_length AS MaxDepth)
                                SELECT 1 FROM const
                                    WHERE x.depth < const.MaxDepth - 1
                            )
                    )
                SELECT * from cte
        )
        SELECT x.id, y.content, x.depth
            FROM
                tree_inheritance AS x INNER JOIN trajectories AS y
                    ON x.id = y.id
            ORDER BY depth DESC
        """
    # same as above, but remove the subquery which limits depth
    query_unlimited = """
        WITH tree_inheritance AS (
                WITH RECURSIVE cte (id, child_id, depth) AS (
                    SELECT :z, NULL, 0
                    UNION ALL
                    SELECT id, child_id, 1
                        FROM tree WHERE tree.child_id = :z
                    UNION
                    SELECT y.id, y.child_id, depth + 1
                        FROM cte AS x INNER JOIN tree AS y ON y.child_id = x.id
                    )
                SELECT * from cte
        )
        SELECT x.id, y.content, x.depth
            FROM
                tree_inheritance AS x INNER JOIN trajectories AS y
                    ON x.id = y.id
            ORDER BY depth DESC
        """
    # same as above, but don't do the inductive step
    query_singleton = """
        WITH tree_inheritance AS (
                WITH cte (id, child_id, depth) AS (
                    SELECT :z, NULL, 0
                    )
                SELECT * from cte
        )
        SELECT x.id, y.content, x.depth
            FROM
                tree_inheritance AS x INNER JOIN trajectories AS y
                    ON x.id = y.id
            ORDER BY depth DESC
        """
    with engine.connect() as conn:
        if trajectory_length == 0:
            return [], [], []
        elif trajectory_length == 1:
            query = query_singleton
        elif trajectory_length == -1:
            query = query_unlimited
        else:
            query = query_depth_limited
        result = conn.execute(
            text(query), {"z": id, "trajectory_length": trajectory_length}
        )

        # persist the result into Python list
        result = result.all()
        ids: List[int] = [w for (w, _, _) in result]
        trajectory = [convert_base64_str_to_bytes(w) for (_, w, _) in result]
        depths: List[int] = [w for (_, _, w) in result]

    return ids, trajectory, depths


class Stepper(Leaf):
    def __init__(self):
        super().__init__()
        self.buffer = []


class FileLinearLocator(Locator):
    """
    Locator which stores branching, linear data
    with minimal redundancies in a SQLite db.

    To use, one should override `deserialize` of
    your Leaf module class to output a serialized Python object `x`
    with `x.buffer` a List type. I.e., any Leaf which uses LinearLocator
    should define an instance variable `buffer` of List type, e.g.,
        ```
        class A(Leaf):

            def __init__(self, buffer = []):
                super().__init__()
                self.buffer = buffer
        ```

    NOTE: This means that the `save_leaf` recursion will not recurse into
        the buffer, and any Leaf types inside will serialize naively.
    """

    def __init__(self, resource_uri: str = ""):
        self.resource_uri = resource_uri
        self.parent_id = -1

    def store(self, bin: bytes, parent_id: int = -1) -> "LeafUID":
        """
        Stores the bin as a child node of the node given by parent_id.
        #### Returns:
        - leaf_uid (LeafUID): formatted path indicating the DB UID and the
                              SQLite unique key corresponding
                              to the inserted node
        """
        # default setting if not set at function call
        if (self.parent_id != -1) and (parent_id == -1):
            parent_id = self.parent_id

        db_name, data_bin = self.parse_bin(bin)

        # create subfolder if not exist
        subdir = os.path.join(self.resource_uri, db_name)
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        # store metadata binary
        # TODO: this is redundant if the subfolder already exists,
        #       as db_name uniquely corresponds to this part of the data
        store_metadata(subdir=subdir, data_bin=data_bin)

        # store data binary
        row_id = store_data(subdir=subdir, data_bin=data_bin, parent_id=parent_id)
        # update parent_uid in instance
        self.parent_id = int(row_id)

        # return leaf_uid
        leaf_uid = LeafUID(db_name + ":" + str(row_id))

        return leaf_uid

    def retrieve(self, uid: "LeafUID", length: int = 1) -> bytes:
        """
        Retrieve trajectory of given length of saved data starting from
        the leaf node given by uid, then traversing backwards towards the root.
        NOTE: length=0 corresponds to the entire trajectory, and it is not
              recommended to unpack large trajectories in memory
        #### Returns:
        - bin (bytes): trajectory packed as a python object x with
                       x.buffer being the array of data
        """
        try:
            db_name, row_id = uid.split(":")
            # set parent_id from retrieval
            self.parent_id = int(row_id)
        # check in case too many strings are returned
        except ValueError:
            raise ValueError("leaf_uid is not properly formatted.")

        db_url = self._db_name_to_db_url(db_name)
        bin = retrieve_trajectory(db_url=db_url, row_id=self.parent_id, length=length)

        return bin

    def _db_name_to_db_url(self, db_name: str) -> str:
        db_url = os.path.join(self.resource_uri, db_name, "lineardb")
        return db_url

    @classmethod
    def parse_bin(cls, bin: bytes) -> Tuple[str, bytes]:
        if bin[20:24] != bytes.fromhex("deadbeef"):
            raise ValueError("Parsed bin is corrupted.")
        else:
            return bin[0:20].hex(), bin[24:]

    @classmethod
    def parse_leaf_uid(cls, uid: LeafUID) -> Tuple[str, int]:
        db_name, node_id = uid.split(":")
        return db_name, int(node_id)

    @staticmethod
    def hash(bin: bytes) -> "LeafUID":
        """LinearLocator must use SHA1 hashes"""
        return LeafUID(sha1(bin).hexdigest())

    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class _EngineContext:
    def __init__(self, db_url: str = ""):
        self.db_url = db_url

    def __enter__(self):
        self.engine = create_engine(f"sqlite+pysqlite:///{self.db_url}", echo=True)
        return self.engine

    def __exit__(self, *args):
        self.engine.dispose()
        del self.engine
        return
