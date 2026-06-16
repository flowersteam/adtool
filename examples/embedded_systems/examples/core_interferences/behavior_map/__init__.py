__all__ = ["InterferenceBehaviorMap"]


def __getattr__(name):
    if name == "InterferenceBehaviorMap":
        from .InterferenceBehaviorMap import InterferenceBehaviorMap

        return InterferenceBehaviorMap
    raise AttributeError(name)
