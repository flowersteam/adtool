"""A collection of useful "pure" functions which wrap computational steps, for
convenience.

Essentially identical in functionality to the interface of
`auto_disc.maps.Map`, but wrappers are reusable and do not have a
particular role in the context of the exploration loop. Consequently, wrappers
may be used to do most of the computational work inside a given `Map`,
conceptually similar to the layers of a deep neural network. For example:
``` python
class MyMap(Map):

    def __init__(*args):
        super().__init__()
        self.a = MyWrapper(args[0])
        self.b = MyWrapper(args[1])

    def map(self, input: Dict) -> Dict:
        return self.b.map(self.a.map(input))

    def sample(self) -> Any:
        return self.b.sample()
```
"""
from .IdentityWrapper import IdentityWrapper
from .SaveWrapper import SaveWrapper
from .TransformWrapper import TransformWrapper
from .WrapperPipeline import WrapperPipeline
