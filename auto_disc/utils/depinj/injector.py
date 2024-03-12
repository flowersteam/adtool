def inject(instance, function, overwrite: bool = True) -> None:
    """Runtime method overrider"""

    # check if should overwrite method
    if getattr(instance, function.__name__, None) and (not overwrite):
        raise Exception(
            f"""Object {instance.__name__} 
                already has method {function.__name__}.
                Pass overwrite=True in order to 
                overwrite existing methods."""
        )

    # need __get__ here to bound the method from the class to the right object
    setattr(instance, function.__name__, function.__get__(instance))

    return


def inject_callbacks(cls, callbacks=[], overwrite=True):
    """Runtime dependency injector, takes a class and an array of callables"""

    def raise_callbacks(self, *args, **kwargs):
        for f in callbacks:
            f(self, *args, **kwargs)
        return

    # Overload methods at runtime
    for f in callbacks:
        inject(cls, f, overwrite=overwrite)

    inject(cls, raise_callbacks, overwrite=overwrite)

    return cls
