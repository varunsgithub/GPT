import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,    #Vocabulary-size
    "context_length": 1024, #Context length
    "emb_dim": 768,         #Embedding dimensions
    "n_heads": 12,          #Number of attention heads
    "n_layers": 12,         #Number of layers
    "drop_rate": 0.1,       #Dropout rate
    "qkv_bias": False       #Query-Key-Value bias
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        #Normal initialization of the super class
        super().__init__()
        #
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(               
            *[DummyTransformerBlock(cfg)               
              for _ in range(cfg["n_layers"])]         
        )                                              
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])     
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    
    #The forward method will help pass the inputs into the model
    #the in_idx is essentially the input tensors
    def forward(self, in_idx):
        #Batch size and sequence length is dependent on the inputs shape
        batch_size, seq_len = in_idx.shape
        #Next we create the token and positional embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        #The final embedding is created by combining the token and positional
        #embedding
        x = tok_embeds + pos_embeds
        
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):    
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):     
        return x

class DummyLayerNorm(nn.Module):          
    def __init__(self, normalized_shape, eps=1e-5):    
        super().__init__()

    def forward(self, x):
        return x