class BaseMutator:
    """
    Base class to mute space var
    """

    def __init__(self):
        raise NotImplementedError

    def __call__(self, x, mutate_mask):
        raise NotImplementedError

    def init_shape(self, shape=None):
        self.shape = shape
