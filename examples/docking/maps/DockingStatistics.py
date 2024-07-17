from typing import Dict
import numpy as np
from copy import deepcopy
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from adtool.wrappers.BoxProjector import BoxProjector

from plip.basic import config
from plip.structure.preparation import PDBComplex
from Bio.PDB import PDBParser
from plip.exchange.report import BindingSiteReport

from io import BytesIO

from examples.docking.systems import Docking

def retrieve_plip_interactions(pdb_file):
    protlig = PDBComplex()


    protlig.load_pdb(pdb_file.decode('utf-8'), as_string=True)

    for ligand in protlig.ligands:
        protlig.characterize_complex(ligand)

    key_site = sorted(protlig.interaction_sets.items())
    if len(key_site) == 0:
        return []
    key, site = key_site[0]

    binding_site = BindingSiteReport(site) 

    return getattr(binding_site, "hydrophobic_info")

def atoms_in_bounding_box(pdb_file, center, size):
    """
    List all atom indices in a given bounding box defined by its center and size.
    
    :param pdb_file: Path to the PDB file
    :param center: A tuple of coordinates for the bounding box center (center_x, center_y, center_z)
    :param size: A tuple of sizes along each axis for the bounding box (size_x, size_y, size_z)
    :return: List of atom indices within the bounding box
    """
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    
    # Calculate bounding box coordinates
    xmin = center_x - size_x / 2
    xmax = center_x + size_x / 2
    ymin = center_y - size_y / 2
    ymax = center_y + size_y / 2
    zmin = center_z - size_z / 2
    zmax = center_z + size_z / 2
    
    # Load the PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    
    # Iterate through all atoms and check if they lie within the bounding box
    atom_indices = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # check if it's a carbon atom
                    if atom.element != 'C':
                        continue
                    x, y, z = atom.coord
                    if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                        atom_indices.append(atom.serial_number)
    
    return atom_indices

class DockingStatistics(Leaf):
    """
    Compute statistics on docking output.
    """

    def __init__(
        self,
        system: Docking,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key

        self.system = system

        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

        self.atom_indices = atoms_in_bounding_box(
            self.system.biomolecule,
            (self.system.center_x, self.system.center_y, self.system.center_z),
            (self.system.size_x, self.system.size_y, self.system.size_z),
        )

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)
        pdb_file = intermed_dict[self.premap_key]
        intermed_dict["raw_" + self.premap_key] = pdb_file
        del intermed_dict[self.premap_key]

        interactions = retrieve_plip_interactions(pdb_file)
        stats = self._calc_statistics(interactions, self.atom_indices)

        intermed_dict[self.postmap_key] = stats
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()

    def _calc_statistics(self, interactions, atom_indices) -> np.ndarray:
        dists = [i[6] for i in interactions]
        protcarbonidxs = [i[8] for i in interactions]
        stats = [
            1 / float(dists[protcarbonidxs.index(a)]) if a in protcarbonidxs else 0
            for a in atom_indices
        ]
        return np.array(stats)
