from __future__ import annotations

import typing
from typing import Dict, List, Union


class History:
    def __init__(self, columns: List[str], documents: List[Dict] = []) -> None:
        self._data = {}
        for column in columns:
            self._data[column] = []

        for document in documents:
            self.append(document)

        self._index = -1
        self._len = 0

    def append(self, document: Dict) -> None:
        """
        Append document

        Args:
            document: dict of database element
        """
        assert (
            document.keys() == self._data.keys()
        ), "Columns of new document do not match history's columns"
        for k, v in document.items():
            self._data[k].append(v)
        self._len += 1

    def __iter__(self) -> History:
        return self

    def __next__(self):
        self._index += 1
        if self._index >= len(self._documents):
            self._index = -1
            raise StopIteration
        else:
            return self[self._index]

    def __getitem__(self, item: Union[int, str]) -> typing.Dict:
        if isinstance(item, int):
            if item >= self._len:
                raise IndexError(
                    f"Index out of range (tried fetching the {item}-th while history has only {self._len} elements)."
                )
            result = {}
            for k, v in self._data.items():
                result[k] = v[item]
            return result
        elif isinstance(item, str):
            if not item in self._data.keys():
                raise Exception(f"Unrecognized column name {item}.")
            return self._data[item]
        else:
            raise TypeError(f"History is not subscriptable with {type(item)} objects.")

    def __len__(self):
        return self._len
