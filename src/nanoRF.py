import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import trange

# nanoRF - a simplified nano implementation of a rectified flow diffusion model. Reference Paper: https://arxiv.org/abs/2403.03206. Borrowed some code from https://github.com/tinygrad/tinygrad/blob/master/examples/minrf.py

# ==========================
# Helpers
# ==========================
def precompute_freqs_cis_torch(dim: int, end: int, theta: float = 10000.0, device: torch.device = None) -> torch.Tensor:
    """
    Precomputes frequencies for rotary embeddings.
    Args:
        dim (int): Dimension of the embeddings.
        end (int): The maximum sequence length.
        theta (float): Frequency scaling factor.
        device (torch.device): Device to store the tensor on.
    Returns:
        torch.Tensor: Precomputed frequencies.
    """
    if device is None:
        device = torch.device("cpu")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.stack([freqs_cos, freqs_sin], dim=-1)

def apply_rotary_emb_torch(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary embeddings to the input tensor.
    Args:
        x (torch.Tensor): Input tensor of shape (B, T, C).
        freqs_cis (torch.Tensor): Precomputed frequencies of shape (T, 2).
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x0 = x_reshaped[..., 0]
    x1 = x_reshaped[..., 1]
    f0 = freqs_cis[..., 0]
    f1 = freqs_cis[..., 1]
    y0 = x0 * f0 - x1 * f1
    y1 = x0 * f1 + x1 * f0
    y = torch.stack([y0, y1], dim=-1).flatten(start_dim=-2)
    return y.type_as(x)

def modulate(x:torch.Tensor, shift:torch.Tensor, scale:torch.Tensor) -> torch.Tensor:
    """
    Modulates the input tensor with shift and scale.
    Args:
        x (torch.Tensor): Input tensor.
        shift (torch.Tensor): Shift tensor.
        scale (torch.Tensor): Scale tensor.
    Returns:
        torch.Tensor: Modulated tensor.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    A simple MLP for timestep embedding.
    Args:
        hidden_size (int): Size of the hidden layer.
    Returns:
        torch.Tensor: Embedded timestep.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    def __call__(self, t:torch.Tensor):
        return self.mlp(t.reshape(-1, 1).float())

# ==========================
# DiT-LLaMA Model
# ==========================
class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = apply_rotary_emb_torch(xq, freqs_cis)
        xk = apply_rotary_emb_torch(xk, freqs_cis)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# LLaMA-style FeedForward (SwiGLU)
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim_multiplier: int = 4, multiple_of: int = 256):
        super().__init__()
        _ff_expanded_dim = dim * hidden_dim_multiplier
        true_hidden_dim = int(2 * _ff_expanded_dim / 3)
        true_hidden_dim = multiple_of * ((true_hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, true_hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, true_hidden_dim, bias=False)
        self.w2 = nn.Linear(true_hidden_dim, dim, bias=False)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps=1e-5):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim_multiplier=4)
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Linear(dim, 6 * dim, bias=True)

    def __call__(self, x:torch.Tensor, freqs_cis:torch.Tensor, adaln_input:torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(F.silu(adaln_input)).chunk(6, dim=1)
        h_attn = modulate(self.attention_norm(x), shift_msa, scale_msa)
        h_attn = self.attention(h_attn, freqs_cis)
        x = x + gate_msa.unsqueeze(1) * h_attn
        h_ffn = modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
        h_ffn = self.feed_forward(h_ffn)
        x = x + gate_mlp.unsqueeze(1) * h_ffn
        return x.contiguous()

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Linear(dim, 2 * dim, bias=True)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
    def __call__(self, x:torch.Tensor, c:torch.Tensor):
        shift, scale = self.adaLN_modulation(F.silu(c)).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

class DiT_Llama(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_layers=6, n_heads=4, num_classes=10, patch_size=2,
                 img_size=32, max_seq_len=512, device: torch.device = None):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.n_heads = n_heads
        self.device = device if device is not None else torch.device("cpu")

        conv_out_channels = dim // 2
        gn_num_groups = 1
        if conv_out_channels > 0:
             gn_num_groups = 32 if (conv_out_channels >= 32 and conv_out_channels % 32 == 0) else 1
        else:
            conv_out_channels = 1 # at least 1 channel for Conv2d

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, conv_out_channels, kernel_size=5, padding=2, stride=1), nn.SiLU(),
            nn.GroupNorm(gn_num_groups, conv_out_channels),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=5, padding=2, stride=1), nn.SiLU(),
            nn.GroupNorm(gn_num_groups, conv_out_channels),
        )
        self.x_embedder = nn.Linear(self.patch_size * self.patch_size * conv_out_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = nn.Embedding(num_classes + 1, dim)
        self.final_layer = FinalLayer(dim, self.patch_size, self.out_channels)
        head_dim = dim // n_heads
        num_patches = (img_size // patch_size)**2
        actual_max_seq_len = max(num_patches, max_seq_len if max_seq_len is not None else 0)
        if actual_max_seq_len == 0: actual_max_seq_len = 256
        freqs_cis_data = precompute_freqs_cis_torch(head_dim, actual_max_seq_len, device=self.device)
        self.register_buffer("freqs_cis", freqs_cis_data)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])

    def unpatchify(self, x: torch.Tensor):
        B, N, _ = x.shape
        P = self.patch_size
        C_out = self.out_channels
        H_patches = W_patches = int(N**0.5)
        assert N == H_patches * W_patches, f"Number of patches {N} must allow integer square root."
        x = x.view(B, H_patches, W_patches, P, P, C_out)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C_out, H_patches * P, W_patches * P)
        return x

    def patchify(self, x: torch.Tensor):
        B, C_in, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"Image dimensions ({H},{W}) must be divisible by patch size {P}."
        num_patches_h = H // P
        num_patches_w = W // P
        x = x.view(B, C_in, num_patches_h, P, num_patches_w, P)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(start_dim=3)
        x = x.flatten(start_dim=1, end_dim=2)
        return x

    def __call__(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.init_conv_seq(x)
        x = self.patchify(x)
        x = self.x_embedder(x)
        adaln_input = self.t_embedder(t) + self.y_embedder(y.long())
        current_seq_len = x.size(1)
        freqs_cis_for_block = self.freqs_cis[:current_seq_len, :]
        for layer in self.layers:
            x = layer(x, freqs_cis_for_block, adaln_input=adaln_input)
        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)
        return x

    def rf(self, x_clean: torch.Tensor, cond_labels: torch.Tensor):
        b = x_clean.shape[0]
        t_rand = torch.randn(b, device=x_clean.device)
        t = torch.sigmoid(t_rand)
        texp = t.view(b, *([1] * (x_clean.ndim - 1)))
        dropout_prob = 0.1
        uncond_token_idx = self.num_classes
        is_dropped = torch.rand(cond_labels.shape[0], device=x_clean.device) < dropout_prob
        eff_cond_labels = torch.where(is_dropped,
                                      torch.tensor(uncond_token_idx, device=x_clean.device, dtype=torch.long),
                                      cond_labels.long())
        z1_noise = torch.randn_like(x_clean)
        zt_interpolated = (1 - texp) * x_clean + texp * z1_noise
        vtheta_predicted = self(zt_interpolated, t, eff_cond_labels)
        target_vector = z1_noise - x_clean
        loss = F.mse_loss(vtheta_predicted, target_vector)
        return loss

    @torch.no_grad()
    def sample(self, z_init: torch.Tensor, cond_labels: torch.Tensor, null_cond_idx: int, sample_steps=50, cfg_scale=2.0):
        self.eval()
        b = z_init.size(0)
        null_cond_labels = torch.full_like(cond_labels, null_cond_idx, dtype=torch.long)
        dt_val = 1.0 / sample_steps
        dt_tensor = torch.full((b, *([1] * (z_init.ndim - 1))), dt_val, device=z_init.device)
        sampled_images_history = [z_init.clone().cpu()]
        z_current = z_init.clone()
        for i in range(sample_steps, 0, -1):
            t_val = i / sample_steps
            t_current_tensor = torch.full((b,), t_val, device=z_init.device)
            v_cond = self(z_current, t_current_tensor, cond_labels.long())
            v_uncond = self(z_current, t_current_tensor, null_cond_labels)
            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
            z_current = z_current - dt_tensor * v_cfg
            z_current = z_current.contiguous()
            # Store only final for brevity, or could store intermediates
        sampled_images_history.append(z_current.clone().cpu()) 
        self.train()
        return sampled_images_history # Returns a list, last element is the final image

# ASCII Visualization 
def mviz_pytorch(t: torch.Tensor, target_individual_size: int = 12, gap_pixels: int = 1):
    """
    Visualizes a batch of images in ASCII with downsampling.
    t: Tensor of shape (B, 1, H, W), expected range [-1, 1].
    target_individual_size: The height and width to downsample each image to.
    """
    assert len(t.shape) == 4 and t.shape[1] == 1, "Input for mviz must be (B, 1, H, W)"
    t = t.cpu()

    B = t.shape[0]
    processed_images_and_gaps = []

    for i in range(B):
        img_original_dim = t[i:i+1, :, :, :]

        # Downsample
        img_downsampled = F.interpolate(
            img_original_dim,
            size=(target_individual_size, target_individual_size),
            mode='bilinear',
            align_corners=False
        )
        img_processed = (img_downsampled.squeeze(0).squeeze(0) + 1) / 2
        img_processed = img_processed.clamp(0, 1)
        processed_images_and_gaps.append(img_processed)
        
        if i < B - 1:
            gap_tensor = torch.zeros(target_individual_size, gap_pixels)
            processed_images_and_gaps.append(gap_tensor)

    if not processed_images_and_gaps: 
        print("No images to visualize.")
        return
    ft_combined_images = torch.cat(processed_images_and_gaps, dim=1) 
    for row_pixels in ft_combined_images.tolist():
        ln = [f"\033[38;5;{232 + int(pixel_val * 23)}m██" for pixel_val in row_pixels]
        print(''.join(ln) + "\033[0m") 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATCH_SIZE = 4
    BATCH_SIZE = 64
    STEPS = 2000
    SAMPLE_STEPS = 20
    OVERFIT = False
    IMG_SIZE = 32
    MODEL_DIM = 64
    N_LAYERS = 6
    N_HEADS = 4
    NUM_CLASSES = 10
    LEARNING_RATE = 5e-4

    print("\n--- nanoRF Configs & Hyperparams ---")
    print(f"Device: {device}")
    print(f"Model Dim: {MODEL_DIM}")
    print(f"Model N Layers: {N_LAYERS}")
    print(f"Model N Heads: {N_HEADS}")
    print(f"Patch Size: {PATCH_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Training Steps: {STEPS}")
    print(f"Sampling Steps: {SAMPLE_STEPS}")
    print(f"Overfit Mode: {OVERFIT}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Num Classes: {NUM_CLASSES}")
    print("-" * 23)

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
    if OVERFIT:
        subset_indices = list(range(min(BS, len(train_dataset_full))))
        train_dataset_effective = torch.utils.data.Subset(train_dataset_full, subset_indices)
    else:
        train_dataset_effective = train_dataset_full
    train_loader = torch.utils.data.DataLoader(
        train_dataset_effective, batch_size=BS, shuffle=not OVERFIT,
        num_workers=0, pin_memory=True if device.type == 'cuda' else False
    )

    model = DiT_Llama(patch_size=PATCH_SIZE, device=device, img_size=IMG_SIZE,
                      dim=MODEL_DIM, n_layers=N_LAYERS, n_heads=N_HEADS, num_classes=NUM_CLASSES).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("-" * 23 + "\n")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data_iter = iter(train_loader)

    try:
        for step in (t_bar := trange(STEPS)):
            model.train()
            try:
                X_batch, Y_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                X_batch, Y_batch = next(data_iter)
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            loss = model.rf(X_batch, Y_batch)
            loss.backward()
            optimizer.step()
            t_bar.set_description(f"loss: {loss.item():.4f}")

            if step % 10 == 0 or step == STEPS - 1:
                print(f"\nStep {step}: Visualizing samples for digits 5, 0, 4")
                z_sample_noise = torch.randn(3, 1, IMG_SIZE, IMG_SIZE, device=device)
                cond_sample_labels = torch.tensor([5, 0, 4], device=device, dtype=torch.long)
                sampled_images_history = model.sample(
                    z_sample_noise, cond_sample_labels, null_cond_idx=model.num_classes,
                    sample_steps=SAMPLE_STEPS, cfg_scale=2.0
                )
                final_sampled_images = sampled_images_history[-1]
                mviz_pytorch(final_sampled_images)
    except Exception as e:
        print(f"Training stopped early due to error: {e}")
    finally:
        print(f"Training loop exited at step {step if 'step' in locals() else 'unknown'}")
