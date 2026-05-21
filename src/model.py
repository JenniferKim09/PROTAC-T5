import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
class ConditionalT5(nn.Module): 
    def __init__(self, t5_model, latent_dim=768): 
        super().__init__() 
        self.t5 = t5_model 
        self.latent_to_emb = nn.Linear(latent_dim, t5_model.config.d_model) 
        
    def forward(self, input_ids, attention_mask, latent_cond, labels=None): 
        cond_emb = self.latent_to_emb(latent_cond).unsqueeze(1) 
        token_emb = self.t5.encoder.embed_tokens(input_ids) 
        inputs_embeds = torch.cat([cond_emb, token_emb], dim=1) 
        cond_mask = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device, 
                               dtype=attention_mask.dtype) 
        new_attention_mask = torch.cat([cond_mask, attention_mask], dim=1) 
        outputs = self.t5(inputs_embeds=inputs_embeds, 
                          attention_mask=new_attention_mask, 
                          labels=labels, use_cache=False) 
        return outputs 

class T5WithContrastive(nn.Module): 
    def __init__(self, pretrain_path, tokenizer, lambda_contrastive=0.5, 
                 temperature=0.07, hidden_size=768): 
        super().__init__() 
        self.t5_model = T5ForConditionalGeneration.from_pretrained(pretrain_path) 
        self.t5_model.resize_token_embeddings(len(tokenizer)) 
        self.encoder = self.t5_model.get_encoder() 
        self.decoder = ConditionalT5(self.t5_model) 
        self.ecloud_encoder = Conv3DEncoder(d_model=hidden_size) 
        self.ecloud_projection = nn.Linear(hidden_size, hidden_size) 
        self.encoder_projection = nn.Linear(self.t5_model.config.d_model, hidden_size) 
        self.classify_head = nn.Linear(self.t5_model.config.d_model, 2)
        self.lambda_contrastive = lambda_contrastive 
        self.temperature = temperature 

    def info_nce_loss(self, rep1, rep2): 
        if torch.isnan(rep1).any() or torch.isnan(rep2).any(): 
            raise RuntimeError("NaN found in reps before normalize") 
        rep1 = nn.functional.normalize(rep1, dim=-1) 
        rep2 = nn.functional.normalize(rep2, dim=-1) 
        logits = torch.matmul(rep1, rep2.T) / max(self.temperature, 1e-3) 
        labels = torch.arange(rep1.size(0)).to(rep1.device) 
        if torch.isnan(logits).any() or torch.isinf(logits).any(): 
            raise RuntimeError("NaN/Inf in contrastive logits") 
        return (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.T, labels)) / 2
    
    def forward(self, eclouds, input_ids, labels): 
        attention_mask = (input_ids != 0).long() 
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask) 
        cls_hidden = output.last_hidden_state[:,0,:] 
        rep = self.encoder_projection(cls_hidden) 
        rep_eclouds = self.ecloud_projection(self.ecloud_encoder(eclouds)) 
        contrastive_loss = self.info_nce_loss(rep, rep_eclouds) 
        outputs = self.decoder(input_ids, attention_mask, rep_eclouds, labels=labels) 
        return outputs, contrastive_loss 
    
    def generate_from_eclouds(self, eclouds, input_ids): 
        attention_mask = (input_ids != 0).long() 
        cond_mask = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device, 
                               dtype=attention_mask.dtype) 
        new_attention_mask = torch.cat([cond_mask, attention_mask], dim=1) 
        rep_eclouds = self.ecloud_projection(self.ecloud_encoder(eclouds)) 
        cond_emb = self.decoder.latent_to_emb(rep_eclouds).unsqueeze(1) 
        token_emb = self.decoder.t5.encoder.embed_tokens(input_ids) 
        inputs_embeds = torch.cat([cond_emb, token_emb], dim=1) 
        encoder_outputs = self.t5_model.encoder(
                            inputs_embeds=inputs_embeds,
                            attention_mask=new_attention_mask,
                            return_dict=True)
        outputs = self.t5_model.generate(encoder_outputs=encoder_outputs, 
                                         attention_mask=new_attention_mask, 
                                         max_length=100, do_sample=True,  
                                         top_p=0.8, # nucleus sampling 
                                         temperature=1.3, 
                                         repetition_penalty=1.0, 
                                         num_beams=1, # beam search 
                                         early_stopping=True ) 
        return outputs
    
    
class Conv3DEncoder(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, d_model=256):
        super(Conv3DEncoder, self).__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv3d(in_channels, d_model // 4, kernel_size, stride, padding, dilation)
        self.conv2 = nn.Conv3d(d_model // 4, d_model // 2, kernel_size, stride, padding, dilation)
        self.conv3 = nn.Conv3d(d_model// 2, d_model, kernel_size, stride, padding, dilation)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        bz = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(bz, -1, self.d_model)
        x = x.mean(dim=1)

        return x