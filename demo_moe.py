#!/usr/bin/env python3
"""
MoE LLM Experiment with Different Gating Strategies

This example builds a very small Transformer language model with one MoE feedforward layer.
The MoE layer uses a custom CUDA kernel (via block_sparse_matmul_autograd) to compute only
the experts selected by the gating network. Two gating strategies are provided:
  - "topk": for each token, select the top-k experts.
  - "sigmoid": use sigmoid activations and a threshold to decide which experts to compute.

This toy example uses synthetic data so that it can run on an NVIDIA 3080.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

# ---------------------------
# Set random seeds for reproducibility.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# Import your custom CUDA kernels.
# (Make sure these are built and installed in your Python environment.)
from block_sparse_matmul import block_sparse_matmul, block_sparse_matmul_backward

# ---------------------------
# Custom autograd function wrapping the block-sparse CUDA kernels.
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

# ---------------------------
# MoE Layer with configurable gating.
# We assume the model dimension (d_model) equals the tile size.
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, gating_type='topk', top_k=1, threshold=0.5):
        """
        d_model: model dimension (must equal tile_size)
        num_experts: number of experts
        gating_type: "topk" or "sigmoid"
        top_k: for topk gating, number of experts to select (default 1)
        threshold: for sigmoid gating, threshold value (default 0.5)
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.gating_type = gating_type
        self.top_k = top_k
        self.threshold = threshold
        # Expert weight: shape (d_model, num_experts*d_model)
        self.weight = nn.Parameter(torch.randn(d_model, num_experts * d_model) * 0.01)
        # Gating network: maps input (d_model) to num_experts.
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: (batch, d_model)
        gate_logits = self.gate(x)  # shape: (batch, num_experts)
        if self.gating_type == 'topk':
            gating_weights = torch.softmax(gate_logits, dim=-1)
            top_values, top_indices = torch.topk(gating_weights, k=self.top_k, dim=-1)
            mask = torch.zeros_like(gating_weights)
            mask.scatter_(1, top_indices, 1)
        elif self.gating_type == 'sigmoid':
            gating_probs = torch.sigmoid(gate_logits)
            gating_weights = gating_probs  # You might optionally renormalize.
            mask = (gating_probs > self.threshold).float()
        else:
            raise ValueError("Invalid gating_type. Choose 'topk' or 'sigmoid'.")

        # --- Pad the batch so its size is divisible by d_model (tile size) ---
        batch = x.size(0)
        remainder = batch % self.d_model
        if remainder != 0:
            pad_size = self.d_model - remainder
            x_padded = torch.cat([x, torch.zeros(pad_size, self.d_model, device=x.device)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_size, self.num_experts, device=x.device)], dim=0)
            gating_weights = torch.cat([gating_weights, torch.zeros(pad_size, self.num_experts, device=x.device)], dim=0)
        else:
            x_padded = x
        new_batch = x_padded.size(0)
        row_blocks = new_batch // self.d_model  # each block has d_model tokens

        # --- Create block mask: for each block, combine token masks with OR (max) ---
        block_mask = mask.view(row_blocks, self.d_model, self.num_experts).max(dim=1)[0]
        block_mask = block_mask.flatten().to(x.device, dtype=torch.int32)

        # --- Call the custom block-sparse multiplication ---
        # x_padded: (new_batch, d_model)
        # self.weight: (d_model, num_experts*d_model)
        out = block_sparse_matmul_autograd(x_padded, self.weight, block_mask, self.d_model)
        # out: (new_batch, num_experts*d_model) --> reshape to (new_batch, num_experts, d_model)
        out = out.view(new_batch, self.num_experts, self.d_model)
        # Combine expert outputs weighted by the original gating weights.
        gating_weights = gating_weights.unsqueeze(-1)  # (new_batch, num_experts, 1)
        out = (gating_weights * out).sum(dim=1)  # (new_batch, d_model)
        # Remove padded tokens.
        out = out[:batch]
        return out

# ---------------------------
# A simple Transformer block that uses the MoE layer as the feedforward.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_experts, gating_type='topk', top_k=1, threshold=0.5, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.moe = MoELayer(d_model, num_experts, gating_type, top_k, threshold)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # Flatten tokens for MoE: reshape to (batch*seq_len, d_model)
        seq_len, batch, d_model = x.size()
        x_flat = x.view(seq_len * batch, d_model)
        moe_out = self.moe(x_flat)
        moe_out = moe_out.view(seq_len, batch, d_model)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)
        return x

# ---------------------------
# A simple Transformer language model using the MoE Transformer block.
class MoETransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, num_experts,
                 gating_type='topk', top_k=1, threshold=0.5, dropout=0.1, max_seq_len=32):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_experts, gating_type, top_k, threshold, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.size()
        emb = self.embed(x)  # (batch, seq_len, d_model)
        emb = emb + self.pos_embed[:, :seq_len, :]
        # Transformer expects (seq_len, batch, d_model)
        out = emb.transpose(0, 1)
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        out = out.transpose(0, 1)  # (batch, seq_len, d_model)
        logits = self.head(out)
        return logits

# ---------------------------
# Synthetic dataset generator (toy language modeling data).
def generate_synthetic_dataset(num_samples, seq_len, vocab_size, device):
    data = torch.randint(low=0, high=vocab_size, size=(num_samples, seq_len), device=device)
    # For language modeling, input and target are offset by one.
    inputs = data[:, :-1]
    targets = data[:, 1:]
    return TensorDataset(inputs, targets)

# ---------------------------
# Training loop for language modeling.
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # model output: (batch, seq_len, vocab_size)
        logits = model(inputs)
        # reshape for loss: (batch*seq_len, vocab_size)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

# ---------------------------
# Main experiment function.
def run_experiment(gating_type='topk', top_k=1, threshold=0.5, num_epochs=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running MoE Transformer LM with gating_type = {gating_type} on {device}")

    # Hyperparameters
    vocab_size = 1000
    d_model = 16            # Must equal tile size for our custom kernel.
    num_layers = 2
    num_heads = 2
    num_experts = 4
    max_seq_len = 32

    model = MoETransformerLM(vocab_size, d_model, num_layers, num_heads, num_experts,
                             gating_type=gating_type, top_k=top_k, threshold=threshold,
                             dropout=0.1, max_seq_len=max_seq_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Generate synthetic dataset.
    num_samples_train = 1024
    num_samples_val = 256
    dataset_train = generate_synthetic_dataset(num_samples_train, max_seq_len, vocab_size, device)
    dataset_val = generate_synthetic_dataset(num_samples_val, max_seq_len, vocab_size, device)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    train_losses = []
    val_losses = []
    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        t_elapsed = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {t_elapsed:.2f}s")
    return train_losses, val_losses

# ---------------------------
# Main entry point.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoE LLM Experiment with Custom Gating")
    parser.add_argument('--gating_type', type=str, default='topk', choices=['topk', 'sigmoid'],
                        help="Gating type: 'topk' or 'sigmoid'")
    parser.add_argument('--top_k', type=int, default=1, help="For topk gating, number of experts to select")
    parser.add_argument('--threshold', type=float, default=0.5, help="For sigmoid gating, threshold value")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    train_losses, val_losses = run_experiment(gating_type=args.gating_type,
                                              top_k=args.top_k,
                                              threshold=args.threshold,
                                              num_epochs=args.epochs,
                                              batch_size=args.batch_size)

    # Plot training and validation losses.
    epochs = np.arange(1, args.epochs+1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"MoE Transformer LM ({args.gating_type} gating)")
    plt.legend()
    plt.show()
