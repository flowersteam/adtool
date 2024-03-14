from adtool.utils.leaf.locators.locators import BlobLocator, LinearLocator


def test_BlobLocator___init__():
    locator = BlobLocator(resource_uri="test")
    assert locator.resource_uri == "test"


def test_BlobLocator_store_FileLocator(mocker):
    resource_uri = "/tmp/locator_test"
    bin = bytes(1)
    locator = BlobLocator(resource_uri=resource_uri)

    # use pytest to mock the internal locator.store() method
    mocker.patch(
        "adtool.utils.leaf.locators.Locator.FileLocator.store",
        return_value="uid_test",
    )

    uid = locator.store(bin)

    assert uid == "uid_test"


def test_BlobLocator_store_ExpeDBLocator(mocker):
    resource_uri = "http://localhost"
    bin = bytes(1)
    locator = BlobLocator(resource_uri=resource_uri)

    # use pytest to mock the internal locator.store() method
    mocker.patch(
        "adtool.utils.leafutils.leafintegrations.expedb_locators.ExpeDBLocator.store",
        return_value="uid_test",
    )

    uid = locator.store(bin)

    assert uid == "uid_test"


def test_BlobLocator_retrieve_FileLocator(mocker):
    resource_uri = "/tmp/locator_test"
    bin = bytes(1)
    locator = BlobLocator(resource_uri=resource_uri)

    # use pytest to mock the internal locator.retrieve() method
    mocker.patch(
        "adtool.utils.leaf.locators.Locator.FileLocator.retrieve", return_value=bin
    )

    retrieved_bin = locator.retrieve("uid_test")

    assert retrieved_bin == bin


def test_BlobLocator_retrieve_ExpeDBLocator(mocker):
    resource_uri = "http://localhost"
    bin = bytes(1)
    locator = BlobLocator(resource_uri=resource_uri)

    # use pytest to mock the internal locator.retrieve() method
    mocker.patch(
        "adtool.utils.leafutils.leafintegrations.expedb_locators.ExpeDBLocator.retrieve",
        return_value=bin,
    )

    retrieved_bin = locator.retrieve("uid_test")

    assert retrieved_bin == bin


def test_LinearLocator___init__():
    locator = LinearLocator(resource_uri="test")
    assert locator.resource_uri == "test"


def test_LinearLocator_store_FileLinearLocator(mocker):
    resource_uri = "/tmp/locator_test"
    bin = bytes(1)
    locator = LinearLocator(resource_uri=resource_uri)

    def mock_store(self, bin, parent_id):
        # update parent_id after storage
        self.parent_id = 13
        return "uid_test"

    # mock the LinearLocator store method
    mocker.patch(
        "adtool.utils.leaf.locators.LinearBase.FileLinearLocator.store",
        new=mock_store,
    )

    uid = locator.store(bin)

    assert uid == "uid_test"  # check return
    assert locator.parent_id == 13  # check parent_id update side effect


def test_LinearLocator_store_ExpeDBLinearLocator(mocker):
    resource_uri = "http://localhost"
    bin = bytes(1)
    locator = LinearLocator(resource_uri=resource_uri)

    def mock_store(self, bin, parent_id):
        # update parent_id after storage
        self.parent_id = 13
        return "uid_test"

    # mock the LinearLocator store method
    mocker.patch(
        "adtool.utils.leafutils.leafintegrations.expedb_locators.ExpeDBLinearLocator.store",
        new=mock_store,
    )

    uid = locator.store(bin)

    assert uid == "uid_test"  # check return
    assert locator.parent_id == 13  # check parent_id update side effect


def test_LinearLocator_retrieve_FileLinearLocator(mocker):
    resource_uri = "/tmp/locator_test"
    bin = bytes(1)
    locator = LinearLocator(resource_uri=resource_uri)

    def mock_retrieve(self, uid, length):
        self.parent_id = 13
        if uid == "uid_test":
            return bin
        else:
            raise Exception("unreachable")

    # mock the LinearLocator retrieve method
    mocker.patch(
        "adtool.utils.leaf.locators.LinearBase.FileLinearLocator.retrieve",
        new=mock_retrieve,
    )

    retrieved_bin = locator.retrieve("uid_test")
    assert retrieved_bin == bin
    assert locator.parent_id == 13


def test_LinearLocator_retrieve_ExpeDBLinearLocator(mocker):
    resource_uri = "http://localhost"
    bin = bytes(1)
    locator = LinearLocator(resource_uri=resource_uri)

    def mock_retrieve(self, uid, length):
        self.parent_id = 13
        if uid == "uid_test":
            return bin
        else:
            raise Exception("unreachable")

    # mock the LinearLocator retrieve method
    mocker.patch(
        "adtool.utils.leafutils.leafintegrations.expedb_locators.ExpeDBLinearLocator.retrieve",
        new=mock_retrieve,
    )

    retrieved_bin = locator.retrieve("uid_test")
    assert retrieved_bin == bin
    assert locator.parent_id == 13
