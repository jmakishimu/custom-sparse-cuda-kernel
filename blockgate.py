#!/usr/bin/env python3
"""
Compare Dense vs. Sparse MoE (with multiple gating strategies) in a small Transformer.
Now includes:
  - Gating: "quadratic" and "pioneer" (cos^2).
  - Print epoch progress with train_loss & val_loss each epoch.

Author: YourName
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Your custom CUDA block-sparse kernels must be installed:
from block_sparse_matmul import block_sparse_matmul, block_sparse_matmul_backward

SAVE_FOLDER = "moe_transformer_experiment"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# -----------------------------
# 1) Synthetic LLM dataset
# -----------------------------
def generate_synthetic_lm_data(num_samples=10000, seq_len=16, vocab_size=50, seed=42):
    """
    Generate random integer tokens for 'input' and shift them by 1 for 'target'
    => a toy language modeling scenario.
    """
    rng = np.random.RandomState(seed)
    inputs = rng.randint(0, vocab_size, size=(num_samples, seq_len))
    # Next token => shift left by 1, random for last
    targets = np.roll(inputs, shift=-1, axis=1)
    for i in range(num_samples):
        targets[i, -1] = rng.randint(0, vocab_size)
    inputs_t = torch.tensor(inputs, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)
    return inputs_t, targets_t

# -----------------------------
# 2) Sparse Kernel Wrappers
# -----------------------------
class BlockSparseMatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, block_mask, tile_size):
        ctx.save_for_backward(A, B, block_mask)
        ctx.tile_size = tile_size
        return block_sparse_matmul(A, B, block_mask, tile_size)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, block_mask = ctx.saved_tensors
        tile_size = ctx.tile_size
        dA, dB = block_sparse_matmul_backward(A, grad_output, B, block_mask, tile_size)
        return dA, dB, None, None

def block_sparse_matmul_autograd(A, B, block_mask, tile_size):
    return BlockSparseMatMulFunction.apply(A, B, block_mask, tile_size)

# -----------------------------
# 3) Additional Gating Strategies
# -----------------------------
def sparsemax(logits):
    """
    2D sparsemax, shape => (batch, K).
    """
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_logits, dim=-1)
    r = torch.arange(1, logits.size(1) + 1, device=logits.device, dtype=logits.dtype).view(1, -1)
    r = r.expand(logits.size(0), -1)
    rhs = 1 + r * sorted_logits
    is_pos = (rhs > cumsum).to(torch.int32)
    k_star = torch.max(is_pos * r, dim=1)[0]
    k_star = torch.clamp(k_star, min=1)
    k_star_int = (k_star - 1).long()
    cumsum_star = torch.gather(cumsum, 1, k_star_int.unsqueeze(1)).squeeze(1)
    tau = (cumsum_star - 1) / k_star
    tau = tau.unsqueeze(1)
    p_sorted = torch.clamp(sorted_logits - tau, min=0)
    p = torch.zeros_like(p_sorted)
    p.scatter_(1, sorted_idx, p_sorted)
    return p

def compute_gating_probs(logits, gating_type="softmax", top_k=2, threshold=0.5):
    """
    Return gating_probs shape (batch, K).
    Gating types:
      - "softmax"
      - "sparsemax"
      - "sigmoid"
      - "topk"
      - "quadratic"
      - "pioneer" (cos^2)
    """
    if gating_type == "softmax":
        return torch.softmax(logits, dim=-1)
    elif gating_type == "sparsemax":
        return sparsemax(logits)
    elif gating_type == "sigmoid":
        return torch.sigmoid(logits)
    elif gating_type == "topk":
        # We'll do normal softmax, but interpret top-k in the logic
        return torch.softmax(logits, dim=-1)
    elif gating_type == "quadratic":
        # (logits^2) => softmax
        quad = logits**2
        return torch.softmax(quad, dim=-1)
    elif gating_type == "pioneer":
        # "invented" gating => cos^2(logits), then softmax
        cos_vals = torch.cos(logits)
        cos_sq = cos_vals * cos_vals
        return torch.softmax(cos_sq, dim=-1)
    else:
        # fallback
        return torch.softmax(logits, dim=-1)

# -----------------------------
# 4) Dense MoE
# -----------------------------
class DenseMoE(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)
        # Each expert is (d_model -> d_model)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4*d_model),
                nn.ReLU(),
                nn.Linear(4*d_model, d_model),
            ) for _ in range(num_experts)
        ])

    def forward(self, x, gating_type="softmax", top_k=1, threshold=0.5):
        logits = self.gate(x)  # (B, K)
        probs = compute_gating_probs(logits, gating_type, top_k, threshold)

        # Additional step if gating_type is "topk", "sigmoid", etc. to produce a final distribution
        if gating_type == "topk":
            topvals, topinds = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topinds, 1.0)
            denom = topvals.sum(dim=1, keepdim=True)
            denom = torch.where(denom==0, torch.ones_like(denom), denom)
            probs = mask * (1.0/denom)
        elif gating_type == "sigmoid":
            mask = (probs>threshold).float()
            sum_mask = mask.sum(dim=1, keepdim=True)
            mask = torch.where(sum_mask==0, torch.ones_like(mask)*(1/mask.size(1)), mask)
            denom = mask.sum(dim=1, keepdim=True)
            probs = mask/denom
        elif gating_type == "softmax":
            # threshold approach
            if threshold < 1.0:
                mask = (probs>threshold).float()
                sum_mask = mask.sum(dim=1, keepdim=True)
                mask = torch.where(sum_mask==0, torch.ones_like(mask)*(1/mask.size(1)), mask)
                denom = mask.sum(dim=1, keepdim=True)
                probs = mask/denom
        elif gating_type == "sparsemax":
            # zero entries => ignoring?
            pass
        elif gating_type == "quadratic":
            # since we returned a softmax over (logits^2), we might want a top_k or threshold too
            # but let's keep it simple for demonstration: no additional step
            pass
        elif gating_type == "pioneer":
            # similarly, it's already in softmax form
            pass

        # Compute experts
        out_experts = []
        for i, exp in enumerate(self.experts):
            out_experts.append(exp(x))  # shape (B, d_model)
        stack_out = torch.stack(out_experts, dim=1) # (B, K, d_model)
        out = (probs.unsqueeze(-1) * stack_out).sum(dim=1)
        return out

# -----------------------------
# 5) Block-Sparse MoE
# -----------------------------
class SparseMoE(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(d_model, num_experts*d_model)*0.01)
        self.gate = nn.Linear(d_model, num_experts)
        self.tile_size = d_model

    def forward(self, x, gating_type="topk", top_k=1, threshold=0.5):
        B = x.size(0)
        logits = self.gate(x)
        probs = compute_gating_probs(logits, gating_type, top_k, threshold)

        # Build binary mask
        if gating_type == "topk":
            topvals, topinds = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topinds, 1.0)
        elif gating_type == "sigmoid":
            mask = (probs>threshold).float()
            sum_mask = mask.sum(dim=1, keepdim=True)
            mask = torch.where(sum_mask==0, torch.ones_like(mask)*(1/mask.size(1)), mask)
        elif gating_type == "softmax":
            # threshold
            mask = (probs>threshold).float()
            sum_mask = mask.sum(dim=1, keepdim=True)
            mask = torch.where(sum_mask==0, torch.ones_like(mask)*(1/mask.size(1)), mask)
        elif gating_type == "sparsemax":
            mask = (probs>0).float()
            sum_mask = mask.sum(dim=1, keepdim=True)
            mask = torch.where(sum_mask==0, torch.ones_like(mask)*(1/mask.size(1)), mask)
        elif gating_type == "quadratic":
            # Already a "softmax" over (logits^2), we can do top_k or threshold
            # For demonstration, let's do a threshold
            mask = (probs>threshold).float()
            sum_mask = mask.sum(dim=1, keepdim=True)
            mask = torch.where(sum_mask==0, torch.ones_like(mask)*(1/mask.size(1)), mask)
        elif gating_type == "pioneer":
            # cos^2 => softmax => threshold
            mask = (probs>threshold).float()
            sum_mask = mask.sum(dim=1, keepdim=True)
            mask = torch.where(sum_mask==0, torch.ones_like(mask)*(1/mask.size(1)), mask)
        else:
            # default top-1
            topvals, topinds = torch.topk(probs, 1, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topinds, 1.0)

        # Pad
        remainder = B % self.tile_size
        if remainder!=0:
            pad_size = self.tile_size - remainder
            x_pad = torch.zeros(pad_size, self.d_model, device=x.device, dtype=x.dtype)
            m_pad = torch.zeros(pad_size, self.num_experts, device=x.device, dtype=mask.dtype)
            x_padded = torch.cat([x, x_pad], dim=0)
            mask_padded = torch.cat([mask, m_pad], dim=0)
            new_B = x_padded.size(0)
        else:
            x_padded = x
            mask_padded = mask
            new_B = B

        row_blocks = new_B // self.tile_size
        # mask_block => (row_blocks, tile_size, num_experts) => max => (row_blocks, num_experts)
        mask_block = mask_padded.view(row_blocks, self.tile_size, self.num_experts).max(dim=1)[0]
        mask_block = mask_block.flatten().to(x.device, dtype=torch.int32)

        # block-sparse matmul
        out_all = block_sparse_matmul_autograd(x_padded, self.weight, mask_block, self.tile_size)
        out_all = out_all.view(new_B, self.num_experts, self.d_model)
        out_all = out_all[:B]

        # Weighted sum
        out = (out_all * probs.unsqueeze(-1)).sum(dim=1)
        return out

# -----------------------------
# 6) Tiny Transformer with MoE
# -----------------------------
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_experts, dense_or_sparse="dense"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dense_or_sparse = dense_or_sparse
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        if dense_or_sparse=="dense":
            self.moe = DenseMoE(d_model, num_experts)
        else:
            self.moe = SparseMoE(d_model, num_experts)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, gating_type="softmax", top_k=1, threshold=0.5):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        seq_len, batch, d_model = x.size()
        x_flat = x.view(seq_len*batch, d_model)
        moe_out = self.moe(x_flat, gating_type, top_k, threshold)
        moe_out = moe_out.view(seq_len, batch, d_model)
        x = x + self.dropout(moe_out)
        x = self.ln2(x)
        return x

class MoETransformerLM(nn.Module):
    def __init__(self, d_model, n_layers, num_heads, num_experts, vocab_size, dense_or_sparse="dense"):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, num_heads, num_experts, dense_or_sparse)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, gating_type="softmax", top_k=1, threshold=0.5):
        batch, seq_len = x.size()
        emb = self.embed(x) + self.pos_embed[:, :seq_len, :]
        hidden = emb.transpose(0,1)
        for blk in self.blocks:
            hidden = blk(hidden, gating_type, top_k, threshold)
        hidden = self.ln_final(hidden)
        hidden = hidden.transpose(0,1)
        logits = self.head(hidden)
        return logits

# -----------------------------
# 7) Training Utilities
# -----------------------------
def cross_entropy_loss(outputs, targets):
    B, T, V = outputs.size()
    return nn.functional.cross_entropy(outputs.view(B*T, V), targets.view(B*T))

def evaluate_model(model, data_loader, gating_type, top_k, threshold, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x, gating_type, top_k, threshold)
            loss = cross_entropy_loss(logits, batch_y)
            num_toks = batch_x.numel()
            total_loss += loss.item() * num_toks
            total_tokens += num_toks
    return total_loss / total_tokens

def evaluate_accuracy(model, data_loader, gating_type, top_k, threshold, device):
    """
    Token-level accuracy: predicted token = argmax(logits).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x, gating_type, top_k, threshold)
            preds = logits.argmax(dim=-1)  # (B, T)
            eqs = (preds == batch_y).float()
            correct += eqs.sum().item()
            total += batch_y.numel()
    return correct / total

def train_one_run(seed, config, device):
    """
    Trains a single run with the given seed & config.
    Returns logs: { train_loss: [...], val_loss: [...], test_acc: float, time: ..., mem_bytes: ... }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(seed)

    # Data
    train_inputs, train_targets = generate_synthetic_lm_data(
        num_samples=config["train_samples"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        seed=seed
    )
    val_inputs, val_targets = generate_synthetic_lm_data(
        num_samples=config["val_samples"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        seed=seed+100
    )
    test_inputs, test_targets = generate_synthetic_lm_data(
        num_samples=config["test_samples"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        seed=seed+200
    )
    train_ds = TensorDataset(train_inputs, train_targets)
    val_ds   = TensorDataset(val_inputs,   val_targets)
    test_ds  = TensorDataset(test_inputs,  test_targets)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"], shuffle=False)

    # Model
    model = MoETransformerLM(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        num_heads=config["num_heads"],
        num_experts=config["num_experts"],
        vocab_size=config["vocab_size"],
        dense_or_sparse=config["dense_or_sparse"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    logs = {
        "train_loss": [],
        "val_loss": [],
        "test_acc": 0.0,
        "time": 0.0,
        "mem_bytes": 0
    }

    gating_type = config["gating_type"]
    top_k = config["top_k"]
    threshold = config["threshold"]

    start_time = time.time()
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()

    # Train loop
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        total_toks = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, gating_type, top_k, threshold)
            loss = cross_entropy_loss(logits, batch_y)
            loss.backward()
            optimizer.step()
            ntoks = batch_x.numel()
            epoch_loss += loss.item() * ntoks
            total_toks += ntoks

        train_epoch_loss = epoch_loss / total_toks
        logs["train_loss"].append(train_epoch_loss)

        # validation
        val_loss = evaluate_model(model, val_loader, gating_type, top_k, threshold, device)
        logs["val_loss"].append(val_loss)

        # Print epoch progress
        print(f"[Seed={seed}] Epoch {epoch+1}/{config['epochs']} => "
              f"train_loss={train_epoch_loss:.4f}, val_loss={val_loss:.4f}")

    # after training => test accuracy
    test_acc = evaluate_accuracy(model, test_loader, gating_type, top_k, threshold, device)
    logs["test_acc"] = test_acc

    elapsed = time.time() - start_time
    logs["time"] = elapsed
    if device.type=="cuda":
        logs["mem_bytes"] = torch.cuda.max_memory_allocated()

    return logs

# -----------------------------
# 8) Multi-run with Configs
# -----------------------------
def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config_template = {
        "train_samples": 40000,
        "val_samples": 5000,
        "test_samples": 5000,
        "seq_len": 16,
        "vocab_size": 100,
        "batch_size": 64,
        "d_model": 32,
        "n_layers": 2,
        "num_heads": 2,
        "num_experts": 15,
        "lr": 1e-3,
        "epochs": 40,
        # gating
        "gating_type": "softmax",
        "top_k": 5,
        "threshold": 0.3,
        # mode
        "dense_or_sparse": "dense"
    }

    # Additional gating including "quadratic" & "pioneer"
    gating_types = ["softmax", "topk"]
    seeds = [101, 23, 245, 254, 231, 667, 977, 998]

    runs = []
    for gating in gating_types:
        for mode in ["dense", "sparse"]:
            cfg = config_template.copy()
            cfg["gating_type"] = gating
            cfg["dense_or_sparse"] = mode
            if gating=="topk":
                cfg["top_k"] = 2
            runs.append(cfg)

    results = {}
    for cfg in runs:
        key = (cfg["gating_type"], cfg["dense_or_sparse"])
        all_logs = []
        for s in seeds:
            log_data = train_one_run(s, cfg, device)
            all_logs.append(log_data)
        results[key] = all_logs
    return results, seeds

# -----------------------------
# 9) Plot & Summaries
# -----------------------------
def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=95):
    data = np.array(data)
    n = len(data)
    boot_means = []
    for i in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_means.append(data[idx].mean())
    boot_means = np.array(boot_means)
    lowp = (100-ci)/2
    highp = 100 - lowp
    mean_ = data.mean()
    low_ = np.percentile(boot_means, lowp)
    high_ = np.percentile(boot_means, highp)
    return mean_, (low_, high_)

def plot_experiment_results(results, seeds):
    import matplotlib
    matplotlib.use('Agg')  # in case no GUI
    # Each key => list of logs

    # 1) Train Loss
    plt.figure(figsize=(10,6))
    for key, logs_list in results.items():
        all_trains = []
        for lg in logs_list:
            all_trains.append(lg["train_loss"])
        arr = np.array(all_trains) # shape (n_seeds, epochs)
        mean_ = arr.mean(axis=0)
        sem_  = arr.std(axis=0)/np.sqrt(arr.shape[0])
        ep = np.arange(1, len(mean_)+1)
        plt.plot(ep, mean_, marker='o', label=str(key))
        plt.fill_between(ep, mean_-sem_, mean_+sem_, alpha=0.2)
    plt.title("Training Loss (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "train_loss.png"))
    plt.close()

    # 2) Validation Loss
    plt.figure(figsize=(10,6))
    for key, logs_list in results.items():
        all_vals = []
        for lg in logs_list:
            all_vals.append(lg["val_loss"])
        arr = np.array(all_vals)
        mean_ = arr.mean(axis=0)
        sem_  = arr.std(axis=0)/np.sqrt(arr.shape[0])
        ep = np.arange(1, len(mean_)+1)
        plt.plot(ep, mean_, marker='o', label=str(key))
        plt.fill_between(ep, mean_-sem_, mean_+sem_, alpha=0.2)
    plt.title("Validation Loss (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "val_loss.png"))
    plt.close()

    # 3) Training Time
    plt.figure(figsize=(10,6))
    keys_list = list(results.keys())
    times_list = []
    for k in keys_list:
        arr_ = [lg["time"] for lg in results[k]]
        times_list.append(arr_)
    means_t = []
    errs_low_t = []
    errs_high_t = []
    for arr in times_list:
        m, (lo,hi) = bootstrap_confidence_interval(arr, 1000, 95)
        means_t.append(m)
        errs_low_t.append(m-lo)
        errs_high_t.append(hi-m)
    xind = np.arange(len(keys_list))
    plt.bar(xind, means_t, yerr=[errs_low_t, errs_high_t], alpha=0.7, capsize=5)
    plt.xticks(xind, [str(k) for k in keys_list], rotation=45, ha='right')
    plt.title("Training Time (s) with 95% CI")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "train_times.png"))
    plt.close()

    # 4) Memory usage
    plt.figure(figsize=(10,6))
    mem_list = []
    for k in keys_list:
        arr_ = [lg["mem_bytes"] for lg in results[k]]
        mem_list.append(arr_)
    means_m = []
    errs_low_m = []
    errs_high_m = []
    for arr in mem_list:
        m, (lo,hi) = bootstrap_confidence_interval(arr, 1000, 95)
        means_m.append(m)
        errs_low_m.append(m-lo)
        errs_high_m.append(hi-m)
    xind = np.arange(len(keys_list))
    plt.bar(xind, means_m, yerr=[errs_low_m, errs_high_m], alpha=0.7, capsize=5)
    plt.xticks(xind, [str(k) for k in keys_list], rotation=45, ha='right')
    plt.title("Peak GPU Memory (bytes) with 95% CI")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "mem_usage.png"))
    plt.close()

    # 5) Test Accuracy
    plt.figure(figsize=(10,6))
    acc_list = []
    for k in keys_list:
        arr_ = [lg["test_acc"] for lg in results[k]]
        acc_list.append(arr_)
    means_a = []
    errs_low_a = []
    errs_high_a = []
    for arr in acc_list:
        m, (lo,hi) = bootstrap_confidence_interval(arr, 1000, 95)
        means_a.append(m)
        errs_low_a.append(m-lo)
        errs_high_a.append(hi-m)
    xind = np.arange(len(keys_list))
    plt.bar(xind, means_a, yerr=[errs_low_a, errs_high_a], alpha=0.7, capsize=5)
    plt.xticks(xind, [str(k) for k in keys_list], rotation=45, ha='right')
    plt.title("Test Accuracy (token-level) with 95% CI")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "test_acc.png"))
    plt.close()

    print("Plots saved in:", SAVE_FOLDER)

    print("\n=== Summary (Time, Mem, TestAcc) ===")
    for i, key in enumerate(keys_list):
        arr_t = times_list[i]
        arr_m = mem_list[i]
        arr_a = acc_list[i]
        mt, (lt, ht) = bootstrap_confidence_interval(arr_t)
        mm, (lm, hm) = bootstrap_confidence_interval(arr_m)
        ma, (la, ha) = bootstrap_confidence_interval(arr_a)
        print(f"{key}: Time mean={mt:.2f}s [CI=({lt:.2f},{ht:.2f})], "
              f"Mem mean={mm/1e6:.2f}MB [CI=({lm/1e6:.2f},{hm/1e6:.2f})], "
              f"TestAcc={ma*100:.2f}% [CI=({la*100:.2f},{ha*100:.2f})]")

    print("\n=== Explanation: Sparse vs. Dense MoE ===")
    print("In a Dense MoE, every expert's output is computed regardless of gating. "
          "This can be slower when many experts exist. In a Sparse MoE, the gating network "
          "produces a binary mask that 'skips' inactive experts via a custom block-sparse kernel. "
          "Hence, if many experts are zeroed out, we do less compute, reducing time & memory.\n"
          "Gating methods (top-k, sparsemax, thresholded softmax/sigmoid, quadratic, pioneer) "
          "select only a subset of experts per sample. This leads to speedups on modern GPUs, "
          "because large parts of the weight matrix are never multiplied.\n"
          "In contrast, a Dense MoE calculates outputs from all experts every time, "
          "increasing compute cost if many experts exist.")

def main():
    results, seeds = run_experiments()
    plot_experiment_results(results, seeds)
    print("Done! Check the plots and summary in", SAVE_FOLDER)

if __name__=="__main__":
    main()
