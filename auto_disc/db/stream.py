from auto_disc.db.get_discoveries_with_filter import _get_discoveries_with_filter


class Stream:
    def __init__(self, initial_filter, size):
        self._indexes = list(range(size))
        self._initial_filter = initial_filter
        self.size = size
        self._index = -1

    def _get(self, index):
        _filter = {"run_idx": index}
        _filter.update(self._initial_filter)
        return _get_discoveries_with_filter(_filter)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get(index)
        elif isinstance(index, slice):
            self._indexes = list(range(self.size))[index]
            return self
        else:
            raise NotImplementedError()

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):
        self._index += 1
        if self._index >= len(self._indexes):
            self._indexes = list(range(self.size))
            raise StopIteration
        else:
            return self._get(self._indexes[self._index])
