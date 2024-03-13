import argparse

from adtool.utils.leaf.tests.test_integration_leaf import DiskLocator, DiskPipeline

if __name__ == "__main__":
    hardcoded = "leaf.tests.test_integration_leaf.DiskPipeline|0981a920897058b10c977ddaf90ee7e125b5e3d3"
    uid = hardcoded

    b = DiskPipeline.load_leaf(uid)
    bool_out = b.l1 == [1, 2, 3, 4]

    if bool_out:
        print("Good!")
        exit(0)
    else:
        print("Bad!")
        exit(1)
