import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 16 # how many independent samples to process at once
block_size = 64 # the number of tokens in the input sequence
max_iters = 5000 # number of iterations to train for
eval_interval = 500 # how often to evaluate the model
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200 # how many iterations to evaluate for
n_embd = 192  # the size of the embedding dimension
n_head = 3 # the number of heads in the multiheadattention models
# head_size = 64, which is standard <- n_embd // n_head
n_layer = 6 # the number of blocks in the model
dropout = 0.2 # the dropout probability
# ------------------------------------------------------------------------------

torch.manual_seed(1337) # for reproducibility

# below is the link to the data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open(r".\NeuralNet\GPT\input.txt", 'r') as f: # for Windows
with open(r"./NeuralNet/GPT/input.txt", 'r') as f: # for codespace
    text = f.read()

# here are all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from chars to ints
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: taking a string and outputting a list of integers
decode = lambda l : ''.join([itos[i] for i in l]) # decoder: take a list of integers and outputs a string

# train and test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # we want the high to be set to len(data) - block_size, to allow for the last target to be assigned
    ix = torch.randint(len(data) - block_size, (batch_size,)) # list of random indices
    x = torch.stack([data[i:i+block_size] for i in ix]) # [for i in ix] get's us the index, from which we extraxt block_size size of list
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval() # model set to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # zeros with len of iterations for which model is to be evaluated
        for k in range(eval_iters):
            X, Y = get_batch(split) # get a random batch of data
            _, loss = model(X, Y)
            losses[k] = loss.item() # storing the loss item
        out[split] = losses.mean() # calculating the mean of the losses
    model.train() # model set to training mode
    return out

# super simple bigram model
torch.manual_seed(1337)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        # compute attention scores ('affinities')
        wei = Q @ K.transpose(-2, -1) * (C ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask out the lower half of the matrix
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)
        # apply the attention to the values
        out = wei @ V # (B, T, head_size)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd is the embedding size, n_head is the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x, targets = None):
        # here we deviate from the original implementation
        # as we have applied the layer norm before the feed forard and self attention block
        # this is a much more common practice these days
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x

class BigramLanguageModel(nn.Module): # inherting from nn.Module
    
    def __init__(self):
        super().__init__() # calling the parent class constructor
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding table, for every token in the vocab, we have an embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # position embedding table, for every position in the block, we have an embedding
        # the above two embeddings are learned during training
        # below is the block Nx, of communication and computation
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)]) # n_layer blocks
        self.ln_f = nn.LayerNorm(n_embd) # layer norm
        # below is the output layer
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to get the logits
        # creating a head 
        # self.sa_head = Head(n_embd) # self attention head
        self.sa_head = MultiHeadAttention(num_heads = 4, head_size = n_embd // 4) # multihead attention head (4 heads, each with n_embd // 4 size
        # feed forward layer
        self.ffwd = FeedForward(n_embd)
        
    
    def forward(self, idx, targets = None):
        B, T = idx.shape # batch_size, and block_size (context size)
        # self.token_embedding_table(idx) returns the embeddings got the token in the context
        # the call behaviour of embedding table is to apply the embedding to the input tensor

        # def __call__(self, idx):
            # self.weight has been initialized with dimension of (vocab_size, n_embd) in the constructor
            # self.out = self.weight[idx]
            # return self.out

        # (B, T) -> idx -> (token encoding) -> self.weight -> embedding 
        # thus the output of the below line is (B, T, token_embedding)
        tok_emb = self.token_embedding_table(idx) # (B, T, token_embedding)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, position_embedding)
        # mind that position_embedding.shape[-1] = token_embedding.shape[-1] in it's value, but the position_embedding is not learned, and the size is kept same for broadcasting purposes
        # holds not just the token embeddings but also the positional embeddings
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_head(x) # (B, T, head_size)
        x = self.ffwd(x) # (B, T, head_size)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # training mode
        if targets is None:
            loss = None
        # evaluation mode
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # this is done for consistency, as the loss function expects a 2D tensor
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get ther predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = 1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))