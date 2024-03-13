def logger(inst, *args, **kwargs):
    print(inst.__dict__)
    for a in args:
        print(a)
    for k, v in kwargs.items():
        print(f"{k}, {v}")
    return


def hw(inst):
    print("hello world")
    return
