
from typing import Dict
import numpy as np
from copy import deepcopy
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from adtool.wrappers.BoxProjector import BoxProjector
from examples.synth.systems.Synth import SynthSimulation


from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import torch


model = Wav2Vec2ForSequenceClassification.from_pretrained("gastonduault/music-classifier")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")

# Function for preprocessing audio for prediction
def preprocess_audio(audio_array):
   # audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

    return feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)


class SynthStatistics(Leaf):
    """
    Compute statistics on Synth's output.
    """

    def __init__(
        self,
        system: SynthSimulation,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key

        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict) -> Dict:
        """
        Compute statistics on Synth's output.
        Args:
            input: Synth's output
        Returns:
            A dictionary with the computed statistics.
        """

        intermed_dict = deepcopy(input)

        # store raw output
        array = np.array(intermed_dict[self.premap_key])
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = array
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(array)

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        # projection= self.projector.sample()
        # # sum to 1
        # projection = projection / np.sum(projection)
        # return projection
        # random dimension

        shape=self.projector.tensor_shape
        projection = np.zeros(shape)
        projection[np.random.randint(0, shape[0])] = 1

        return projection


    

    def _calc_static_statistics(self, array: np.ndarray) -> np.ndarray:




        inputs = preprocess_audio(array)

        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
            # compute probabilities
            probs = logits.softmax(dim=-1).flatten().numpy()

        return probs