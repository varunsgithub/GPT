import torch
import torch.nn as nn
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

GPT_CONFIG_124M = {
    "vocab_size": 50257,    #Vocabulary-size
    "context_length": 1024, #Context length
    "emb_dim": 768,         #Embedding dimensions
    "n_heads": 12,          #Number of attention heads
    "n_layers": 12,         #Number of layers
    "drop_rate": 0.1,       #Dropout rate
    "qkv_bias": False       #Query-Key-Value bias
}

#Integrated multi attention head class :)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        #Initialize the nn.Module super class
        super().__init__()
        
        assert(d_out % num_heads == 0), \
        "d_out must be divisible by the number of heads"

        self.d_out = d_out
        
        self.num_heads = num_heads
        
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out)    #2
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )


    def forward(self, x):
        
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)         #3
        
        queries = self.W_query(x)    #3
        
        values = self.W_value(x)     #3

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)       #4
        
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        
        queries = queries.view(                                             
            b, num_tokens, self.num_heads, self.head_dim                    
        )                                                                   

        
        keys = keys.transpose(1, 2)          #5
        
        queries = queries.transpose(1, 2)    #5
        
        values = values.transpose(1, 2)      #5

        attn_scores = queries @ keys.transpose(2, 3)   #6
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    #7

        attn_scores.masked_fill_(mask_bool, -torch.inf)     #8

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)   #9
 #10
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        
        context_vec = self.out_proj(context_vec)    #11
        
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

#Class for feedforward mech
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                                    GELU(),
                                    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
        
    def forward(self, x):
        return self.layers(x)

#Transformer block class   
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
 #1
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      #2

        shortcut = x         #3
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      #4
        return x
    
#the final GPT Model

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #The positional and token embedding layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        #We create n_layers numbered transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        #The final norm is the final normalization layer
        self.final_norm = LayerNorm(cfg["emb_dim"])
        #The final linear output layer which turns the dimension into the vocab size !
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        #The token embeddings are passed in through the linear layer
        tok_embeds = self.tok_emb(in_idx)

        #Next the positional embeddings are created
        pos_embeds = self.pos_emb(torch.arrange(seq_len, device=in_idx.device))

        #The final token embedding is created with the sum of the two
        x = tok_embeds + pos_embeds

        #Drop is applied
        x = self.drop_emb(x)

        #The token is passed through the transformers
        x = self.trf_blocks(x)

        #The final layer normalization are applied on the embedding
        x = self.final_norm(x)

        #The logits produce the final embedding of the vocab dimension
        logits = self.out_head(x)

        return logits