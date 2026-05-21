import numpy as np
from src.chem import get_mol_center
from src.ecloud_utils.xtb_density import CDCalculator, interplot_ecloud
from src.ecloud_utils.grid import BuildGridCenters
import h5py
from rdkit import Chem
from transformers import T5TokenizerFast
from tqdm import tqdm
def protocol(grid, resolution=0.5):
    '''
    Define the grid protocol, including grid size, resolution, and grid centers
        grid size is the number of grids in each dimension
        resolution is the distance between two grid points

    '''
    N = [grid, grid, grid]
    llc = (np.zeros(3) - float(grid * resolution / 2)) + resolution / 2  # lower left corner
    grids = BuildGridCenters(llc, N, resolution)    

    return {'grids':grids, 'N':N, 'llc':llc}

def get_ligecloud(mol, calculater, protocol):
    '''
    Input:
        mol: rdkit 3D mol
        calculater: xtb density calculater
        protocol: protocol for the grid, format: {'grids':(32, 32, 32, 3), 'N':[32, 32, 32]}
        add_noise: add noise to the ligand grid
    Output:
        lig_density: ligand electron density, shape: (32, 32, 32)
    '''
    stand_grid = protocol['grids']
    N = protocol['N']
    mol_center = get_mol_center(mol) 
    lig_grids = stand_grid + mol_center 
    lig_ecloud = calculater.calculate(mol)
    lig_density = interplot_ecloud(lig_ecloud, lig_grids.transpose(3, 0, 1, 2)).reshape(N)
    return lig_density

def single_process(mol):
    calculater = CDCalculator(xtb_command='/home/jinjieyu/protacgen/src/ecloud_utils/xtb-bleed/bin/xtb') # change to your path
    try:
        ecloud=get_ligecloud(mol, calculater, protocol(grid=64))
        s=Chem.MolToSmiles(mol)
        return [ecloud,s]
    except:
        return None


max_input_len = 128
max_target_len = 64

tokenizer = T5TokenizerFast.from_pretrained("ckpt/tokenizer")
mols = Chem.SDMolSupplier('data/PROTAC.sdf')
with open('data/PROTAC.csv') as f:
    data = [line.strip('\n').split(',') for line in f][1:]
df = {}
for i, inf in enumerate(data):
    df[inf[0]] = [inf[-3],inf[-2],inf[-1],i]

print('all data: ', len(mols))
e = []
input = []
out = []
ids = []
for mol in tqdm(mols):
    mol_id = mol.GetProp('_Name')
    pair=single_process(mol)
    assert pair is not None
    e.append(pair[0])
    row = df[mol_id]
    frag_a = str(row[0]).strip()
    linker = str(row[1]).strip()
    frag_b = str(row[2]).strip()
    src = f"<FRAG_A> {frag_a} <SEP> <FRAG_B> {frag_b}"

    tokened_src = tokenizer(
            src,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )["input_ids"]
    tokened_tgt = tokenizer(
            linker,
            max_length=max_target_len,
            truncation=True,
            padding="max_length",
        )["input_ids"]
    input.append(tokened_src)
    out.append(tokened_tgt)
    ids.append(row[3])

with h5py.File('data/test.h5','w') as f_:
    eclouds=f_.create_dataset("eclouds", (len(e),64,64,64), dtype='f')
    input_ids=f_.create_dataset("input_ids", (len(e),max_input_len), dtype='i')
    labels=f_.create_dataset("labels", (len(e),max_target_len), dtype='i')
    id=f_.create_dataset("ids", (len(e),1), dtype='i')
    for i in range(len(e)):
        eclouds[i]=e[i]
        input_ids[i]=input[i]
        labels[i]=out[i]
        id[i]=ids[i]


