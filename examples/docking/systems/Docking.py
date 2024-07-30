import subprocess
from adtool.utils.expose_config.expose_config import expose
from pydantic import BaseModel, Field
from plip.exchange.report import BindingSiteReport

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import base64
import os
import imageio
import io
import time

from typing import Dict, Any, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# reduce plip logging
import logging
logging.getLogger('plip').setLevel(logging.CRITICAL)
logging.getLogger('selenium').setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('WDM').setLevel(logging.CRITICAL)

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
        input["output"] = open(complex, "rb").read().decode('utf-8')
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:

        options = webdriver.ChromeOptions()
        
        options.add_argument('--headless') 
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        
        ligand =  open('examples/docking/systems/tmp/docked_ligand.pdb').read()
        protein = open(self.biomolecule).read()

        # HTML content
        html_content = f'''
        <html>
        <body style="margin: 0; padding: 0; display: block;">
        <div id="3dmolviewer" style="position: relative; width: 512px; height: 512px;">
        <p id="viewer3dmolwarning" style="background-color:#ffcccc;color:black"><br></p>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.2.0/3Dmol-min.js"></script>
        <script>
        var viewer = null;
        var warn = document.getElementById("viewer3dmolwarning");
        if(warn) {{
            warn.parentNode.removeChild(warn);
        }}
        
        viewer = $3Dmol.createViewer(document.getElementById("3dmolviewer"),{{backgroundColor:"white"}});
        viewer.zoomTo();
        viewer.addModel(`{protein}`,"pdb");
        viewer.addModel(`{ligand}`,"pdb");
        viewer.setStyle({{"model": 0}},{{"cartoon": {{"color": "spectrum", "opacity": 0.7}}}});
        viewer.setStyle({{"model": 1}},{{"stick": {{"colorscheme": "cyanCarbon"}}}});
        viewer.zoomTo({{"model": 1}});
        viewer.zoom(0.4);
        viewer.render();

        const nb_frames = 60;
        var pngURIs = [];
        for (var i = 0; i < nb_frames; i++) {{
            viewer.rotate(360/nb_frames , "y");
            pngURIs.push(viewer.pngURI());
        }}
        document.body.setAttribute('data-png-uris', JSON.stringify(pngURIs));
        document.body.setAttribute('data-render-complete', 'true');
        </script>
        </body>
        </html>
        '''

        with open('temp.html', 'w') as f:
            f.write(html_content)

        driver.get('file://' + os.path.abspath('temp.html'))

 


        png_uris = driver.execute_script("return JSON.parse(document.body.getAttribute('data-png-uris'));")

        png_frames = []
        for uri in png_uris:
            png_data = uri.split(',')[1]
            png_bytes = base64.b64decode(png_data)
            png_frames.append(imageio.v2.imread(png_bytes))

        driver.quit()

        os.remove('temp.html')

        memory_file = io.BytesIO()

        imageio.mimsave(memory_file,
                        png_frames, fps=10, format='mp4')
        

        # create smiles images
        smiles = data_dict["params"]["dynamic_params"]["smiles"]

        #using rdKit to generate the image
        m = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(m, size=(300, 300))

        #save it in memory
        memory_file2 = io.BytesIO()

        img.save(memory_file2, format='PNG')




        return [
            (memory_file.getvalue(), 'mp4'),
            (memory_file2.getvalue(), 'png')
        ]