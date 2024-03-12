import os

from auto_disc.utils.leaf.tests.test_integration_leaf import DiskLocator, DiskPipeline

if __name__ == "__main__":
    a = DiskPipeline()
    os.mkdir("/tmp/PipelineDir")
    os.mkdir("/tmp/ResDir")
    a.save_leaf()
    uid = a.uid
