from __future__ import annotations

import typing
from typing import Type, Union

from adtool.auto_disc.utils.misc.history import History
from tinydb import TinyDB
from tinydb.queries import Query, where
from tinydb.storages import JSONStorage, MemoryStorage


class DB(TinyDB):
    def __init__(self) -> None:
        # super().__init__('./db-cache.json', storage=JSONStorage)
        super().__init__(storage=MemoryStorage)

    def close(self) -> None:
        super().close()

    def to_autodisc_history(
        self, documents: list, keys: typing.List[str], new_keys: typing.List[str] = None
    ) -> History:
        """
        Select only some keys in documents and return a History. Use `new_keys` to rename these keys in the returned History.

        Args:
            documents: list of db item
            keys: list of key
            new_keys: list of key
        Returns:
            History
        """
        result_keys = keys if new_keys is None else new_keys
        assert len(keys) == len(result_keys)
        results = History(result_keys)

        for document in documents:
            current_result = {}
            add_document = True
            for idx, key in enumerate(keys):
                if key in document:
                    current_result[result_keys[idx]] = document[key]
                else:
                    add_document = False

            if add_document:
                results.append(current_result)

        return results

    def __getitem__(self, index: Union[int, slice]) -> None:
        """
        Use indexing and slicing over db.

        Args:
            index: DB index/slice
        """
        if isinstance(index, int):
            if index >= 0:
                return self.search(where("idx") == index)
            else:
                return self.search(where("idx") == len(self) + index)
        elif isinstance(index, slice):
            db_idx = list(range(len(self)))
            return self.search(Query().idx.test(lambda val: val in db_idx[index]))
        else:
            raise NotImplementedError()

    def save(self) -> DB:
        """
        Save DB.
        """
        return self

    def load(self, saved_dict):
        """
        Reload DB.
        """
        pass
