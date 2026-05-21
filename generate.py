import torch
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast
from src.dataset import ecloud_dataset
from src.model import T5WithContrastive
from rdkit import Chem
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)
    # random.seed(seed)

setup_seed(2025)
device = torch.device("cuda:0")
torch.cuda.set_device(device)

tokenizer = T5TokenizerFast.from_pretrained("ckpt/tokenizer")
model = T5WithContrastive("laituan245/molt5-small", tokenizer) 
state_dict = torch.load("ckpt/best_model.pt", map_location="cuda")
model.load_state_dict(state_dict) 
model.to(device)

eval_dataset = ecloud_dataset("data/test.h5", rotate=False)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

linkers = []
model.eval()
with torch.no_grad():
    for batch in eval_loader:
        eclouds = torch.Tensor(batch["eclouds"]).to(torch.float).to(device)
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        id = batch["ids"][0]

        out = model.generate_from_eclouds(
            eclouds, input_ids
        )
        linker = tokenizer.decode(out[0], skip_special_tokens=True)
        if Chem.MolFromSmiles(linker):
            linkers.append([id, linker])

with open('generated_linkers.csv','w') as f:
    for p in linkers:
        f.write(f'{p[0]},{p[1]}\n')