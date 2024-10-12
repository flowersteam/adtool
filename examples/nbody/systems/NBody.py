import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import io

# Global constants
G = 1.0  # Gravitational constant
MASS = 1.0  # Mass of each body
DT = 0.001  # Time step for the simulation
SAVE_INTERVAL = 10  # Save interval for the simulation

@dataclass
class NBodyParams:
    speeds: np.ndarray
    positions: np.ndarray

    # after init method convert to numpy arrays 
    def __post_init__(self):
        self.speeds = np.array(self.speeds)
        self.positions = np.array(self.positions)

class NBodySimulation:
    def __init__(
        self,
        max_steps: int = 3000,
        max_distance_limit: float = 10,
        pattern_size: int = 4000,
        N: int = 3
    ) -> None:
        self.max_steps = max_steps
        self.max_distance_limit = max_distance_limit
        self.pattern_size = pattern_size
        self.N = N
        self.params = None
        self.positions_over_time = None
        self.distances_over_time = None
        self.timestep = None

    def compute_forces(self, positions):
        forces = np.zeros_like(positions)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                diff = positions[j] - positions[i]
                distance = np.linalg.norm(diff)
                if distance > 1e-5:  # To avoid division by zero
                    force_magnitude = G * MASS * MASS / distance**2
                    force_direction = diff / distance
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction
        return forces

    def pairwise_distances(self, positions):
        distances = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                distance = np.linalg.norm(positions[j] - positions[i])
                distances.append(distance)
        return distances

    def run_simulation(self):
        print("Running simulation")
        positions = self.params.positions
        velocities = self.params.speeds

        #substract the mean of the positions
        positions -= np.mean(positions, axis=0)

        #substract the mean of the velocities
        velocities -= np.mean(velocities, axis=0)

        print("Positions shape", positions)
        print("Velocities shape", velocities)
        
        positions_over_time = []
        distances_over_time = []
        saved_step = 0

        
        for step in range(self.max_steps):
            forces = self.compute_forces(positions)
            velocities += forces * DT
            positions += velocities * DT
            
            if max(self.pairwise_distances(positions)) > self.max_distance_limit:
                self.positions_over_time = np.array(positions_over_time)
                self.distances_over_time = np.array(distances_over_time)
                self.timestep = step
                return 
            
            if step % SAVE_INTERVAL == 0:  # Save interval is 10
                saved_step += 1
                if len(positions_over_time) < self.pattern_size:
                    positions_over_time.append(positions.copy())
                    distances_over_time.append(self.pairwise_distances(positions))
                elif saved_step % 2 == 0:
                    positions_over_time = positions_over_time[1:]
                    distances_over_time = distances_over_time[1:]
                    positions_over_time.append(positions.copy())
                    distances_over_time.append(self.pairwise_distances(positions))
        
        self.positions_over_time = np.array(positions_over_time)
        self.distances_over_time = np.array(distances_over_time)
        self.timestep = self.max_steps
        print("ici")

    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        self.params = NBodyParams(**input["params"])
        self.run_simulation()
        input["output"] = {
            "positions": self.positions_over_time,
            "distances": self.distances_over_time,
            "timestep": self.timestep
        }
        print("Simulation complete")
        print(self.distances_over_time.shape)
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        plt.figure(figsize=(10, 10))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'] 
        print(self.positions_over_time.shape)
        for i in range(self.N):
            plt.plot(self.positions_over_time[:, i, 0], 
                     self.positions_over_time[:, i, 1], 
                     label=f'Body {i+1}', 
                     color=colors[i % len(colors)])
            
            # Add an dot at the end of the trajectory
            plt.scatter(self.positions_over_time[-1, i, 0], 
                        self.positions_over_time[-1, i, 1], 
                        color=colors[i % len(colors)])
     #   plt.title(f'N-Body Simulation Trajectories (Steps: {self.timestep})')
     #   plt.xlabel('X Position')
     #   plt.ylabel('Y Position')
       # plt.legend()
       # plt.grid(True)
        plt.tight_layout()
        # hide axes
        plt.axis('off')
        # add a text below with the number of steps
        plt.text(0.5, 0.05, f'Steps: {self.timestep}', ha='center', fontsize=12, transform=plt.gcf().transFigure)
        # add text saying the steps range plotted

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        plt.close()  # Close the figure to free up memory

        return [(buf.getvalue(), "png")]