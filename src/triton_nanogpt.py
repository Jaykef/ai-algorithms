import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
import time

# ================================================================
# Implements custom triton kernels for training Karpathy's nanoGPT. 
# The WHY behind this ordeal?
# After practicing triton for about 2 weeks, I attempted 
# implementing custom Triton kernels for Karpathy's nanoGPT. 
# Still not perfect and would appreciate contributions:)
# Docs: https://triton-lang.org/main/index.html
# Demo: https://x.com/jaykef_/status/1847943239531131338
# ================================================================

# -----------------------------
# Data Prep
# -----------------------------
def dataset(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Dataset downloaded and saved to {filepath}.")
    else:
        print(f"Dataset already exists at {filepath}.")

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filepath = "input.txt"

dataset(url, filepath)
with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(text):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def decode(indices):
    return ''.join([itos[i.item()] for i in indices])

data = encode(text)

n = int(0.9 * len(data))
train_data = data[:n].cuda()
test_data = data[n:].cuda()

print(f"Training data size: {train_data.numel()} characters")
print(f"Testing data size: {test_data.numel()} characters")

# -----------------------------
# Triton Kernels
# -----------------------------

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
    output_row_ptr = output_ptr + row_idx * output_row_stride + col_offsets

    logits = tl.load(input_row_ptr, mask=mask, other=float('-inf'))
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0) + 1e-6

    softmax_output = exp_logits / sum_exp_logits
    tl.store(output_row_ptr, softmax_output, mask=mask)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    N, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_offset = x_ptr + row_idx * N + cols
    x = tl.load(x_offset, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    y = (x_centered * rstd) * w + b
    tl.store(y_ptr + row_idx * N + cols, y, mask=mask)

@triton.jit
def cross_entropy_loss_kernel(
    logits_ptr, targets_ptr, loss_ptr, 
    n_classes, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    targets = tl.load(targets_ptr + offsets, mask=mask, other=-1)

    row_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(n_classes):
        col_offset = offsets * n_classes + i
        logit = tl.load(logits_ptr + col_offset, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, logit)

    loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(n_classes):
        col_offset = offsets * n_classes + i
        logit = tl.load(logits_ptr + col_offset, mask=mask, other=float('-inf'))
        exp_logit = tl.exp(logit - row_max)
        row_sum += exp_logit
        loss = tl.where(targets == i, loss - logit + row_max, loss)

    loss += tl.log(row_sum)

    tl.store(loss_ptr + offsets, loss, mask=mask)

@triton.jit
def gelu_kernel(
    x_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    sqrt_2_over_pi = 0.7978845608028654
    coeff = sqrt_2_over_pi * (1 + 0.044715 * x * x)
    y = 0.5 * x * (1 + (x * coeff) / (1 + tl.abs(x * coeff)))

    tl.store(y_ptr + offsets, y, mask=mask)

# -----------------------------------
# Triton-accelerated Launch Functions
# -----------------------------------

class TritonSoftmax(nn.Module):
    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(-1, original_shape[-1])
        x = x.clamp(-100, 100)
        B, N = x.shape
        y = torch.empty_like(x)
        grid = lambda meta: (B,)
        softmax_kernel[grid](
            y, x,
            x.stride(0), y.stride(0), N,
            BLOCK_SIZE=triton.next_power_of_2(N)
        )
        y = y + 1e-8
        y = y / y.sum(dim=-1, keepdim=True)
        return y.view(original_shape)
    
def triton_cross_entropy_loss(logits, targets):
    return TritonCrossEntropyLoss.apply(logits, targets)

class TritonCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        n_elements, n_classes = logits.shape
        loss = torch.empty(n_elements, device=logits.device, dtype=logits.dtype)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        cross_entropy_loss_kernel[grid](
            logits, targets, loss,
            n_classes, n_elements,
            BLOCK_SIZE=1024
        )
        
        ctx.save_for_backward(logits, targets)
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        batch_size, n_classes = logits.shape

        logits_exp = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
        softmax_output = logits_exp / logits_exp.sum(dim=-1, keepdim=True)

        grad_input = softmax_output.clone()
        grad_input.scatter_add_(1, targets.unsqueeze(1), -torch.ones_like(grad_input))
        grad_input *= grad_output.view(-1, 1) / batch_size

        return grad_input, None


class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = eps

    def forward(self, x):
        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape, "Input shape does not match normalized_shape."
        y = torch.empty_like(x)
        x_ = x.reshape(-1, self.normalized_shape[-1])
        y_ = y.reshape(-1, self.normalized_shape[-1])
        M, N = x_.shape
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
        layer_norm_kernel[grid](
            x_, self.weight, self.bias, y_,
            N, eps=self.eps,
            BLOCK_SIZE=128
        )
        return y

class TritonGELU(nn.Module):
    def forward(self, x):
        n_elements = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_kernel[grid](
            x, y, n_elements,
            BLOCK_SIZE=1024
        )
        return y

# -----------------------------
# Model
# -----------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, seq_length, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.seq_length = seq_length

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.softmax = TritonSoftmax()
        self.register_buffer("mask", torch.tril(torch.ones(seq_length, seq_length)).bool())

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~self.mask[:T, :T], float('-inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            TritonGELU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, seq_length, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, seq_length, dropout)
        self.ff = FeedForward(dim, 4 * dim, dropout)
        self.ln1 = TritonLayerNorm(dim)
        self.ln2 = TritonLayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, seq_length, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(seq_length, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, seq_length, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = TritonLayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.seq_length, f"Input sequence length {T} exceeds model's maximum sequence length {self.seq_length}"

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def compute_loss(self, logits, targets):
        return triton_cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

#----------------------------
# Training
#----------------------------

def train(model, train_data, val_data, batch_size, seq_length, learning_rate, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - seq_length, (batch_size,))
        x = torch.stack([data[i:i+seq_length] for i in ix])
        y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
        return x.to(model.token_embedding.weight.device), y.to(model.token_embedding.weight.device)

    def estimate_mfu(model, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in model.parameters())
        L, H, Q, T = model.num_layers, model.num_heads, model.dim // model.num_heads, model.seq_length
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T * batch_size  # multiply by batch size
        flops_achieved = flops_per_fwdbwd * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    iter_num = 0
    best_val_loss = float('inf')
    val_losses = []

    model.train()
    t0 = time.time()
    for epoch in range(num_epochs):
        for _ in range(100):  # 100 batches per epoch
            iter_num += 1

            t_start = time.time()

            # Data loading
            xb, yb = get_batch('train')
            t_data = time.time()

            # Forward pass
            logits = model(xb)
            t_forward = time.time()

            # Loss computation
            loss = model.compute_loss(logits, yb)
            t_loss = time.time()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN or Inf detected in loss at iteration {iter_num}")
                print(f"Logits min: {logits.min()}, max: {logits.max()}")
                print(f"Target min: {yb.min()}, max: {yb.max()}")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            t_backward = time.time()

            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            t_optim = time.time()

            if iter_num % 10 == 0:
                dt = t_optim - t_start
                dt_data = t_data - t_start
                dt_forward = t_forward - t_data
                dt_loss = t_loss - t_forward
                dt_backward = t_backward - t_loss
                dt_optim = t_optim - t_backward
                mfu = estimate_mfu(model, dt)
                
                print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%")
                # print(f"  Data loading: {dt_data*1000:.2f}ms")
                # print(f"  Forward pass: {dt_forward*1000:.2f}ms")
                # print(f"  Loss computation: {dt_loss*1000:.2f}ms")
                # print(f"  Backward pass: {dt_backward*1000:.2f}ms")
                # print(f"  Optimizer step: {dt_optim*1000:.2f}ms")
                # print(f"  Other time: {(dt - dt_data - dt_forward - dt_loss - dt_backward - dt_optim)*1000:.2f}ms")

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(50):  # 50 val batches
                xb, yb = get_batch('val')
                logits = model(xb)
                val_loss += model.compute_loss(logits, yb).item()
        val_loss /= 50
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'Checkpoints/nanoGPT_cpkt.pth')
            print(f"Saved checkpoint for validation loss: {best_val_loss:.4f}")

        model.train()

    return model, val_losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    vocab_size = 65
    dim = 384
    num_heads = 6
    num_layers = 6
    seq_length = 256
    dropout = 0.1
    batch_size = 64
    learning_rate = 3e-4
    num_epochs = 500

    model = NanoGPT(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_length=seq_length,
        dropout=dropout
    ).to(device)

    model.config = type('Config', (), {
        'n_layer': num_layers,
        'n_head': num_heads,
        'n_embd': dim,
        'block_size': seq_length
    })

    # Train config
    model, validation_losses = train(
        model,
        train_data,
        test_data,
        batch_size=batch_size,
        seq_length=seq_length,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    # Load checkpoint
    model.load_state_dict(torch.load('checkpoints/nanoGPT_cpkt.pth', weights_only=True))

    # Generate sample
    model.eval()
    start_text = "Once upon"
    input_ids = encode(start_text).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(240):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_logits = torch.clamp(next_token_logits, -100, 100)
            probs = F.softmax(next_token_logits, dim=-1) + 1e-8
            probs = probs / probs.sum()
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_text = decode(input_ids[0].cpu())
    print("Generated Text:")
    print(generated_text)
