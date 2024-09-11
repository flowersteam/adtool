import typing
from copy import deepcopy
from adtool.utils.leaf.Leaf import Leaf
from examples.draft.systems.Draft import Draft
from adtool.wrappers.BoxProjector import BoxProjector


class DraftStatistics(Leaf):
    """
    Outputs System embedding.
    """

    def __init__(
        self,
        system : Draft,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()

        self.premap_key = premap_key
        self.postmap_key = postmap_key


        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)


    def map(self, input: typing.Dict) -> typing.Dict:
        """
        Compute statistics on System output
        Args:
            input: System's output
        Returns:
            Return a torch tensor in dict
        """

        intermed_dict = deepcopy(input)

        # store raw output
        tensor = intermed_dict[self.premap_key]
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        del intermed_dict[self.premap_key]

        embedding = [] # calculate embedding, must be numpy or list
        intermed_dict[self.postmap_key] = embedding
        
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()
