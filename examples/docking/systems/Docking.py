import subprocess
from adtool.utils.expose_config.expose_config import expose
from pydantic import BaseModel, Field
from plip.exchange.report import BindingSiteReport

from typing import Dict, Any, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

# reduce plip logging
import logging
logging.getLogger('plip').setLevel(logging.CRITICAL)

def run_gnina(protein, ligand, docked_ligand_pdb,center_x, center_y, center_z, size_x, size_y, size_z):

    cmd = [
        "./examples/docking/systems/gnina",
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--num_mc_saved", str(1),
        "--num_modes", str(1),
        "--seed", str(42),
        "--autobox_extend", str(1),
        "--exhaustiveness", str(32),
        "--cnn_scoring", "rescore",
        "--verbosity=0",
        "-r", str(protein),
        "-l", str(ligand),
        "--out", docked_ligand_pdb
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return
    # return result.stdout.decode('utf-8')

def read_box_file(box_file):
    with open(box_file) as f:
        lines = f.readlines()
    center_x = float(lines[0].split('=')[1].strip())
    center_y = float(lines[1].split('=')[1].strip())
    center_z = float(lines[2].split('=')[1].strip())
    size_x = float(lines[3].split('=')[1].strip())
    size_y = float(lines[4].split('=')[1].strip())
    size_z = float(lines[5].split('=')[1].strip())

    return center_x, center_y, center_z, size_x, size_y, size_z

def generate_ligand_pdb(smiles, output_file):



    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    # make hydrogens explicit


    AllChem.EmbedMolecule(m)
    AllChem.MMFFOptimizeMolecule(m, maxIters=1000000)
    Chem.MolToPDBFile(m, output_file)

def merge_pdb_files_with_hetatm(ligand_file, protein_file, output_file):
    with open(ligand_file, 'r') as file:
        ligand_content = file.readlines()

    with open(protein_file, 'r') as file:
        protein_content = file.readlines()

    protein_atoms = [line for line in protein_content if line.startswith('ATOM')]
    protein_hetatms = [line for line in protein_content if line.startswith('HETATM')]
    ligand_hetatms = [line for line in ligand_content if line.startswith('HETATM')]

    if not protein_content[0].startswith('MODEL'):
        protein_content.insert(0, 'MODEL        1\n')
        protein_content.append('ENDMDL\n')

    combined_content = protein_atoms + protein_hetatms + ligand_hetatms

    if not combined_content[-1].strip() == 'END':
        combined_content.append('END\n')

    with open(output_file, 'w') as file:
        file.writelines(combined_content)


class GenerationParams(BaseModel):
    biomolecule: str
    bbox: str


@expose
class Docking:
    config=GenerationParams

    def __init__(
        self,
        *args, **kwargs
    ) -> None:

        self.biomolecule = self.config.biomolecule
        self.center_x, self.center_y, self.center_z, self.size_x, self.size_y, self.size_z = read_box_file(self.config.bbox)


    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        smiles = input["params"]["dynamic_params"]["smiles"]
        
        ligand_pdb = "examples/docking/systems/tmp/ligand.pdb"
        docked_ligand_pdb = "examples/docking/systems/tmp/docked_ligand.pdb"
        complex= "examples/docking/systems/tmp/complex.pdb"
        generate_ligand_pdb(smiles, ligand_pdb)
        run_gnina(self.biomolecule, ligand_pdb,docked_ligand_pdb, self.center_x, self.center_y, self.center_z, self.size_x, self.size_y, self.size_z)
        merge_pdb_files_with_hetatm(docked_ligand_pdb, self.biomolecule, complex)
        input["output"] = open(complex, "rb").read()
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        return data_dict["output"], "pdb"
