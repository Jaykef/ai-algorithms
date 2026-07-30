#!/usr/bin/env python3
"""
Lightweight/nanoGPT implementation of KDA (Kimi Delta Attention).
KDA is a variant of the attention mechanism that incorporates delta updates to the key-value memory.

MLA                            KDA
A = softmax(QKᵀ / √d)          S'ₜ = Diag(αₜ)Sₜ₋₁
O = AV                         Sₜ = S'ₜ + βₜkₜ(vₜ - S'ₜᵀkₜ)ᵀ
Compute: O(T²d)                Compute: O(Td²)
Memory:  O(Td)                 Memory:  O(d²)
"""
import math, os, random, time, urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1337
STEPS = 1000
BATCH_SIZE = 8
BLOCK_SIZE = 64
N_LAYER = 2
N_HEAD = 8
N_EMBD = 64
DROPOUT = 0.0
LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_EVERY = 25
EVAL_BATCHES = 10
GENERATE_TOKENS = 300
BENCHMARK_LENGTHS = [32, 64, 128, 256, 512]
BENCHMARK_REPEATS = 5
BENCHMARK_WARMUPS = 2
CPU_THREADS = max(1, min(8, os.cpu_count() or 1))
MAX_CONTEXT = max(BLOCK_SIZE, max(BENCHMARK_LENGTHS))
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DATA_FILE = ROOT / "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def seed_all(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

def load_text():
    if not DATA_FILE.exists():
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return DATA_FILE.read_text(encoding="utf-8")

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * progress)) * (LR - MIN_LR)

def median_ms(fn):
    with torch.inference_mode():
        for _ in range(BENCHMARK_WARMUPS):
            fn()
        times = []
        for _ in range(BENCHMARK_REPEATS):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2]

class PrepData:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        cut = int(0.9 * len(ids))
        self.train, self.val = ids[:cut], ids[cut:]
        self.vocab_size = len(chars)

    def batch(self, split, generator):
        data = self.train if split == "train" else self.val
        starts = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,), generator=generator)
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in starts])
        y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in starts])
        return x, y

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

class SoftmaxAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.out = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, N_HEAD, c // N_HEAD)
        q, k, v = [z.view(shape).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))

    def decode_step(self, x, key_cache, value_cache, position):
        b, _, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, 1, N_HEAD, c // N_HEAD)
        q, k, v = [z.view(shape).transpose(1, 2) for z in (q, k, v)]
        key_cache[:, :, position:position + 1] = k
        value_cache[:, :, position:position + 1] = v
        keys = key_cache[:, :, :position + 1]
        values = value_cache[:, :, :position + 1]
        scores = (q @ keys.transpose(-2, -1)) * q.size(-1) ** -0.5
        weights = F.softmax(scores, dim=-1)
        y = weights @ values
        return self.out(y.transpose(1, 2).contiguous().view(b, 1, c))

class KimiDeltaAttention(nn.Module):
    """Minimal KDA: decay memory, correct its key-value prediction, then read.

    For each token and head:
        S = Diag(alpha) S
        error = v - S^T k
        S = S + beta * k error^T
        output = S^T q
    """
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.alpha = nn.Linear(N_EMBD, N_EMBD)
        self.beta = nn.Linear(N_EMBD, N_HEAD)
        self.out = nn.Linear(N_EMBD, N_EMBD, bias=False)
        nn.init.constant_(self.alpha.bias, 2.0)

    def project(self, x):
        b, t, c = x.shape
        d = c // N_HEAD
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.normalize(q.view(b, t, N_HEAD, d), dim=-1)
        k = F.normalize(k.view(b, t, N_HEAD, d), dim=-1)
        v = v.view(b, t, N_HEAD, d)
        alpha = torch.sigmoid(self.alpha(x)).view(b, t, N_HEAD, d)
        beta = torch.sigmoid(self.beta(x))
        return q, k, v, alpha, beta

    def forward(self, x):
        b, t, c = x.shape
        d = c // N_HEAD
        q, k, v, alpha, beta = self.project(x)
        state = x.new_zeros(b, N_HEAD, d, d)
        outputs = []
        for i in range(t):
            state = state * alpha[:, i].unsqueeze(-1)
            prediction = torch.einsum("bhd,bhdv->bhv", k[:, i], state)
            error = v[:, i] - prediction
            write_key = beta[:, i].unsqueeze(-1) * k[:, i]
            state = state + torch.einsum("bhd,bhv->bhdv", write_key, error)
            outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, i], state))
        y = torch.stack(outputs, dim=1).reshape(b, t, c)
        return self.out(y)

    def decode_step(self, x, state):
        b, _, c = x.shape
        q, k, v, alpha, beta = self.project(x)
        q, k, v = q[:, 0], k[:, 0], v[:, 0]
        alpha, beta = alpha[:, 0], beta[:, 0]
        state = state * alpha.unsqueeze(-1)
        prediction = torch.einsum("bhd,bhdv->bhv", k, state)
        error = v - prediction
        write_key = beta.unsqueeze(-1) * k
        state = state + torch.einsum("bhd,bhv->bhdv", write_key, error)
        y = torch.einsum("bhd,bhdv->bhv", q, state).reshape(b, 1, c)
        return self.out(y), state

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = SoftmaxAttention() if kind == "MLA" else KimiDeltaAttention()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, kind):
        super().__init__()
        self.token = nn.Embedding(vocab_size, N_EMBD)
        self.position = nn.Embedding(MAX_CONTEXT, N_EMBD)
        self.blocks = nn.ModuleList([Block(kind) for _ in range(N_LAYER)])
        self.norm = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.apply(self.init_weights)
        self.head.weight = self.token.weight
        for module in self.modules():
            if isinstance(module, KimiDeltaAttention):
                nn.init.constant_(module.alpha.bias, 2.0)

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        t = idx.size(1)
        positions = torch.arange(t, device=idx.device)
        x = self.token(idx) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1] / 0.8, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

@torch.no_grad()
def evaluate(model, data, seed):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = data.batch("val", generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def train(kind, data):
    seed_all()
    model = NanoGPT(data.vocab_size, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED + 1)
    history, ema = [], None

    for step in range(STEPS):
        start = time.perf_counter()
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data.batch("train", generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        elapsed = time.perf_counter() - start
        tokens_per_second = BATCH_SIZE * BLOCK_SIZE / max(elapsed, 1e-9)
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        val_loss = None

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            val_loss = evaluate(model, data, SEED + 10_000 + step)

        history.append((step, loss.item(), ema, val_loss))
        val_text = f"; val: {val_loss:.4f};" if val_loss is not None else ";"
        print(f"step: {step:03d}; loss: {loss.item():.4f}; ema: {ema:.4f}{val_text} {kind:8s}")

    sample = model.generate(torch.zeros((1, 1), dtype=torch.long), GENERATE_TOKENS)
    (OUT / f"sample_{kind.lower()}.txt").write_text(data.decode(sample[0]), encoding="utf-8")
    torch.save(model.state_dict(), OUT / f"model_{kind.lower()}.pt")
    return history

@torch.no_grad()
def benchmark(vocab_size):
    results = {name: {"MLA": [], "KDA": []} for name in ("kernel", "prefill", "decode")}
    attention = {"MLA": SoftmaxAttention().eval(), "KDA": KimiDeltaAttention().eval()}
    models = {"MLA": NanoGPT(vocab_size, "MLA").eval(), "KDA": NanoGPT(vocab_size, "KDA").eval()}
    d = N_EMBD // N_HEAD

    print("\nMLA vs KDA")
    print("-" * 72)

    for length in BENCHMARK_LENGTHS:
        x = torch.randn(1, length, N_EMBD)
        mla_kernel = median_ms(lambda: attention["MLA"](x))
        kda_kernel = median_ms(lambda: attention["KDA"](x))
        results["kernel"]["MLA"].append(mla_kernel)
        results["kernel"]["KDA"].append(kda_kernel)
        print(f"kernel ; input: {length:4d}; MLA: {mla_kernel:9.4f} ms; KDA: {kda_kernel:9.4f} ms")

        tokens = torch.randint(0, vocab_size, (1, length))
        mla_prefill = median_ms(lambda: models["MLA"](tokens))
        kda_prefill = median_ms(lambda: models["KDA"](tokens))
        results["prefill"]["MLA"].append(mla_prefill)
        results["prefill"]["KDA"].append(kda_prefill)
        print(f"prefill; input: {length:4d}; MLA: {mla_prefill:9.4f} ms; KDA: {kda_prefill:9.4f} ms")

        token = torch.randn(1, 1, N_EMBD)
        keys = torch.randn(1, N_HEAD, length + 1, d)
        values = torch.randn(1, N_HEAD, length + 1, d)
        state = torch.randn(1, N_HEAD, d, d)

        mla_decode = median_ms(lambda: attention["MLA"].decode_step(token, keys, values, length))
        kda_decode = median_ms(lambda: attention["KDA"].decode_step(token, state)[0])
        results["decode"]["MLA"].append(mla_decode)
        results["decode"]["KDA"].append(kda_decode)
        print(f"decode ; input: {length:4d}; MLA: {mla_decode:9.4f} ms; KDA: {kda_decode:9.4f} ms")
        print("-" * 72)

    return results

def plot_training(histories):
    plt.figure(figsize=(7, 4.5))
    for kind in ("MLA", "KDA"):
        steps = [row[0] for row in histories[kind]]
        ema = [row[2] for row in histories[kind]]
        validation = [row for row in histories[kind] if row[3] is not None]
        plt.plot(steps, ema, label=kind, linewidth=2)
        plt.scatter([row[0] for row in validation], [row[3] for row in validation], s=24)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "training_loss.png", dpi=200)
    plt.close()

def plot_efficiency(results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [
        ("kernel", "Input Length", "Execution Time of Kernels (ms)"),
        ("prefill", "Prefilling Length", "Latency (ms)"),
        ("decode", "Decoding Length", "TPOT (ms)")
    ]

    for index, (axis, panel) in enumerate(zip(axes, panels)):
        key, xlabel, ylabel = panel
        axis.plot(BENCHMARK_LENGTHS, results[key]["MLA"], "--*", label="MLA", linewidth=2, markersize=8)
        axis.plot(BENCHMARK_LENGTHS, results[key]["KDA"], "-o", label="KDA", linewidth=2, markersize=5)
        axis.set_xscale("log", base=2)
        axis.set_xticks(BENCHMARK_LENGTHS)
        axis.set_xticklabels(BENCHMARK_LENGTHS)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3, linestyle="--")
        axis.legend()
        axis.text(0.5, -0.22, f"({chr(97 + index)})", transform=axis.transAxes, ha="center", fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(OUT / "efficiency_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    seed_all()
    data = PrepData(load_text())
    histories = {"MLA": train("MLA", data), "KDA": train("KDA", data)}
    plot_training(histories)
    plot_efficiency(benchmark(data.vocab_size))

if __name__ == "__main__":
    main()