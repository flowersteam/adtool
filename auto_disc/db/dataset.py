from auto_disc.db.get_discoveries_with_filter import _get_discoveries_with_filter
from auto_disc.db.stream import Stream


class Dataset:
    def __init__(self, experiment_id, checkpoint_ids=[]):
        self._filter = {"experiment_id": experiment_id}

        if len(checkpoint_ids) > 0:
            self._filter["checkpoint_id"] = {"$in": self.checkpoint_ids}

    def filter(self, _filter):
        _filter.update(self._filter)
        return _get_discoveries_with_filter(_filter)

    @property
    def stream(self):
        return Stream(self._filter, len(self))

    @property
    def values(self):
        return _get_discoveries_with_filter(self._filter)

    def __getitem__(self, index):
        index_filter = None
        if isinstance(index, int):
            if index >= 0:
                index_filter = index
            else:
                index_filter = len(self) + index
        elif isinstance(index, slice):
            dataset_idx = list(range(len(self)))
            index_filter = {"$in": dataset_idx[index]}
        else:
            raise NotImplementedError()

        _filter = {"run_idx": index_filter}

        _filter.update(self._filter)
        result = _get_discoveries_with_filter(_filter)
        return result

    def __len__(self):
        return (
            max(
                [
                    element["run_idx"]
                    for element in _get_discoveries_with_filter(
                        self._filter, {"run_idx": 1}
                    )
                ]
            )
            + 1
        )
