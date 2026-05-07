import io
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np

from adtool.systems.System import System


class BaseEmbeddedSystem(System, ABC):
    """Base system class for embedded systems."""

    @abstractmethod
    def map(self, input: Dict) -> Dict:
        ...

    @abstractmethod
    def render(self, data_dict: Dict[str, Any]) -> List[Tuple[bytes, str]]:
        ...

    def render_fallback(self, width: int = 64, height: int = 64) -> List[Tuple[bytes, str]]:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        byte_img = io.BytesIO()
        imageio.imwrite(byte_img, frame, format="png")
        byte_img.seek(0)
        return [(byte_img.getvalue(), "png")]
