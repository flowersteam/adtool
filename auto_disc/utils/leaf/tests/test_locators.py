import os
import pathlib

from auto_disc.utils.leaf.locators.Locator import FileLocator


def test_FileLocator___init__():
    locator = FileLocator()
    # removed default init of FileLocators
    assert locator.resource_uri == ""


def test_FileLocator_store():
    locator = FileLocator()
    bytestring = b"pasudgfpausdgfpxzucbv"

    uid = locator.store(bytestring)

    save_dir = os.path.join(os.getcwd(), uid)
    save_path = os.path.join(save_dir, "metadata")

    assert os.path.exists(save_path)
    os.remove(save_path)


def test_FileLocator_retrieve():
    bytestring = b"pasudgfpausdgfpxzucbv"
    fake_uid = "abcdefg"

    res_dir = os.getcwd()
    save_dir = os.path.join(res_dir, fake_uid)
    os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "metadata")

    with open(save_path, "wb") as f:
        f.write(bytestring)

    locator = FileLocator(res_dir)
    bin = locator.retrieve(fake_uid)

    assert bin == bytestring

    os.remove(save_path)
    os.rmdir(save_dir)
