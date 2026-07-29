from adtool.utils.leaf.LeafUID import LeafUID
from adtool.utils.leaf.locators.Locator import FileLocator, Locator


class BlobLocator(Locator):
    """Factory locator for regular, content-addressed Leaf blobs."""

    def __init__(self, resource_uri: str = "", *args, **kwargs):
        self.resource_uri = resource_uri

    def store(self, bin: bytes, *args, **kwargs) -> "LeafUID":
        return self._parse_uri(self.resource_uri)(resource_uri=self.resource_uri).store(
            bin, *args, **kwargs
        )

    def retrieve(self, uid: "LeafUID", *args, **kwargs) -> bytes:
        return self._parse_uri(self.resource_uri)(resource_uri=self.resource_uri).retrieve(
            uid, *args, **kwargs
        )

    @staticmethod
    def _parse_uri(resource_uri: str) -> type:
        return FileLocator
