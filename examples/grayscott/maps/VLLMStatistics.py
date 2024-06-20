import os
import typing
import cv2
import base64
import openai
from typing import List
from copy import deepcopy

import torch

from adtool.utils.leaf.Leaf import Leaf
from adtool.wrappers.BoxProjector import BoxProjector
from examples.grayscott.systems.GrayScott import GrayScott

from openai import OpenAI


# Ensure you have your OpenAI API key set up
openai.api_key = 'YOUR_API_KEY'


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class VLLMStatistics(Leaf):
    """
    Outputs embedding based on OpenAI's embedding service.
    """

    def __init__(
        self,
        system: GrayScott,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()

        self.premap_key = premap_key
        self.postmap_key = postmap_key

        self.SX = system.width
        self.SY = system.height

        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: typing.Dict) -> typing.Dict:
        """
        Compute statistics on Lenia's output
        Args:
            input: Lenia's output
            is_output_new_discovery: indicates if it is a new discovery
        Returns:
            Return a torch tensor in dict
        """

        intermed_dict = deepcopy(input)

        # store raw output
        tensor = intermed_dict[self.premap_key].detach().clone()
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        del intermed_dict[self.premap_key]

        # Convert the tensor to a series of frames
        frames = self._convert_tensor_to_frames(tensor)

        # Generate base64 frames
        base64_frames = self._generate_base64_frames(frames)

        # Generate the description using OpenAI's API
        if len(base64_frames) == 0:
            description = "1: blank"
        else:
            description = self._generate_description(base64_frames)

        # Generate the embedding from the description
        embedding = self._get_embedding(description)

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()

    def _calc_distance(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distance between 2 embeddings in the latent space
        /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        return (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

    def _convert_tensor_to_frames(self, tensor: torch.Tensor) -> List:
        """ Convert tensor to a series of frames """
        frames = []
        tensor = tensor.cpu().numpy()
        for i in range(tensor.shape[0]):
            frame = tensor[i]
            frames.append((frame * 255).astype('uint8'))
        return frames

    def _generate_base64_frames(self, frames: List) -> List[str]:
        """ Convert frames to base64 strings """
        # base64_frames = []
        # for frame in frames:
        #     _, buffer = cv2.imencode(".jpg", frame)
        #     base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        # return base64_frames

        #same but only starting at the frame 500 each 500 frames
        base64_frames = []
        for frame in frames[500::500]:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        return base64_frames
    

    def _generate_description(self, base64_frames: List[str]) -> str:
        """ Generate textual description using OpenAI's API """
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    """Here's a series of images spaced one second apart.
With the eye of an expert biologist, mathematician, computer scientist, physicist, chemist, naturalist, historical and contemporary art, associate a lexical field at the same time precise, qualitative and quantitative, describing at high and low level, the following process and intricate patterns.
Don't just say 'complex' or 'detailed', be more precise by using a tailored vocabulary to describe spatial and temporal patterns (even from one frame to the other), and making analogies with real-world objects or phenomena.
Don't talk about colors since it's a grayscale image.
If an image is just a black or blank screen, just say 'black' or 'blank'.
Changes between frames are considered fast. 
Start directly with:
Frame_{number}: {Detailed contextual lexical field}
""",
                    *map(lambda x: {"image": x, "resize": 768}, base64_frames[500::500]
                         ),
                ],
            },
        ]
        params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 1000,
        }

        result = client.chat.completions.create(**params)
        return result.choices[0].message.content

    def _get_embedding(self, text: str, model: str ="text-embedding-3-large") -> torch.Tensor:
        """ Generate embedding from text using OpenAI's API """
        embedding= client.embeddings.create(input = [text], model=model).data[0].embedding
        return torch.tensor(embedding)
