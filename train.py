import os
import torch
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, AdamW
from src.dataset import ecloud_dataset
from src.model import T5WithContrastive
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)
    # random.seed(seed)

def get_lambda(step, max_lambda=0.3):
    warmup_steps = 10 
    if step < warmup_steps:
        return max_lambda * (step / warmup_steps)  
    else:
        return max_lambda
    
setup_seed(2025)
device = torch.device("cuda:0")
torch.cuda.set_device(device)

tokenizer = T5TokenizerFast.from_pretrained("ckpt/tokenizer")
model = T5WithContrastive("laituan245/molt5-small", tokenizer) 
model.to(device)

train_dataset = ecloud_dataset("data/train_t5_final.h5", rotate=True)
eval_dataset = ecloud_dataset("data/valid_t5_final.h5", rotate=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
optimizer = AdamW(model.parameters(), lr=3e-5)
output_dir = "ckpt/test" 
os.makedirs(output_dir, exist_ok=True)
 
gradient_accumulation_steps = 4
global_step = 0
best_val = 1e6
for epoch in range(200):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader)):
        eclouds = torch.Tensor(batch["eclouds"]).to(torch.float).to(device)
        eclouds += (eclouds>0.2).int() * (torch.rand_like(eclouds) * 0.4 - 0.2).to(device)
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device).long()
        out, contrastive_loss = model(
            eclouds, input_ids, labels
        )
        lam = get_lambda(global_step, max_lambda=0.3)
        loss = out.loss + lam * contrastive_loss
        loss = loss / gradient_accumulation_steps
        loss.backward()
        running_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        global_step += 1

    model.eval()
    with torch.no_grad():
        val_losses = 0
        for batch in eval_loader:
            eclouds = torch.Tensor(batch["eclouds"]).to(torch.float).to(device)
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device).long()
            out, contrastive_loss = model(
                eclouds, input_ids, labels
            )
            val_loss = out.loss + lam * contrastive_loss
            val_losses += val_loss
        print(f"Epoch {epoch} train_loss={running_loss / len(train_loader) :.4f} \
              val_loss={val_losses / len(eval_loader):.4f}")

        # early stopping on val loss
        if val_losses < best_val:
            best_val = val_losses
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
            print(f"Saved best model at epoch {epoch+1}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 3:
            print(f"No improvement for 3 epochs. Early stopping.")
            break

print(f"Training finished. Best val loss: {best_val/len(eval_loader):.4f} at epoch {best_epoch}")