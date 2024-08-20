import subprocess
from adtool.utils.expose_config.expose_config import expose
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple, List

class GenerationParams(BaseModel):
    kernel: List[float] = Field(default_factory=lambda: [0.05, 0.2, 0.05, 0.2, 0.0, 0.2, 0.05, 0.2, 0.05])

@expose
class FlashLenia:
    config = GenerationParams

    def __init__(self, *args, **kwargs) -> None:
        self.kernel = self.config.kernel

    def run_flashlenia(self, kernel):
        cmd = ["./examples/flashlenia/systems/flashlenia"] + [str(k) for k in kernel]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr

    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        kernel = input.get("params", {}).get("dynamic_params", {}).get("kernel", self.kernel)
        stdout, stderr = self.run_flashlenia(kernel)
        
        # Parse the output to extract mean and variance of entropy
        lines = stdout.split('\n')
        mean_entropy = float(lines[0].split(': ')[1])
        variance_entropy = float(lines[1].split(': ')[1])
        
        input["output"] = {
            "mean_entropy": mean_entropy,
            "variance_entropy": variance_entropy
        }
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        # This method is left empty as per the original example
        return []