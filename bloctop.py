import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from block_sparse_matmul import block_sparse_matmul, block_sparse_matmul_backward

# =====================================
# 1) Config
# =====================================
SELECTED_GATING_TYPES = [
    "block_sparse_moe",  # top-K block-sparse MoE
]

N_RUNS = 1
EPOCHS = 40
BATCH_SIZE = 64      # Now we can have multiple row-blocks if tile_size=16 => 64/16=4
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_ENTROPY = 0.01

EXPERT_COUNTS = [10] # e.g. 10 experts
SAVE_FOLDER = "Neuro-MoE_experiment_proto"
os.makedirs(SAVE_FOLDER, exist_ok=True)

N_SAMPLES = 40000
INPUT_DIM = 16   # tile_size=16 => each row-block is 16 samples
HIDDEN_DIM = 32
OUTPUT_DIM = 1

TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, count={torch.cuda.device_count()}")

MULTI_GPU = (torch.cuda.device_count() > 1)

# =====================================
# 2) Data Generation
# =====================================
def generate_train_val_test_dataset(seed, test_split=TEST_SPLIT, val_split=VAL_SPLIT):
    """
    We'll set drop_last=True => so every batch is exactly BATCH_SIZE=64
    which is multiple of tile_size=16 => row_blocks=64/16=4
    """
    torch.manual_seed(seed)
    X_data = torch.randn(N_SAMPLES, INPUT_DIM, device=device)
    y_data = torch.sin(X_data.sum(dim=1)) + 0.1 * torch.randn(N_SAMPLES, device=device)

    indices = torch.randperm(N_SAMPLES, device=device)
    test_size = int(test_split*N_SAMPLES)
    test_idx = indices[-test_size:]
    train_val_idx = indices[:-test_size]

    val_size = int(val_split*len(train_val_idx))
    val_idx  = train_val_idx[-val_size:]
    train_idx= train_val_idx[:-val_size]

    X_train, y_train= X_data[train_idx], y_data[train_idx]
    X_val,   y_val  = X_data[val_idx],   y_data[val_idx]
    X_test,  y_test = X_data[test_idx],  y_data[test_idx]

    train_loader= DataLoader(TensorDataset(X_train, y_train),
                             batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader  = DataLoader(TensorDataset(X_val,   y_val),
                             batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(TensorDataset(X_test,  y_test),
                             batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_loader,val_loader,test_loader

# =====================================
# 3) Dynamic Multi-RowBlock Top-K MoE
# =====================================
class DynamicBlockSparseMoE(nn.Module):
    """
    We do top-K gating for the *entire batch*, then replicate that mask
    across all row-blocks.

    If BATCH_SIZE=64, tile_size=16 => row_blocks=4.

    Steps:
     1) gating => shape (batch_size, num_experts)
     2) sum => shape (num_experts)
     3) top-k => shape (num_experts) mask
     4) replicate => [row_blocks, num_experts] => flatten => length=(row_blocks*num_experts)
     5) block_sparse_matmul => shape (batch_size, num_experts*hidden_dim)
     6) aggregator => final => (batch_size, out_dim)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=10,
                 top_k=3, tile_size=16):
        super().__init__()
        self.num_experts= num_experts
        self.hidden_dim = hidden_dim
        self.top_k      = top_k
        self.tile_size  = tile_size
        self.in_dim     = input_dim
        self.out_dim    = output_dim

        # Gating net => (in_dim -> num_experts)
        self.gating_net = nn.Linear(input_dim, num_experts)

        # Weight => (in_dim, num_experts*hidden_dim)
        total_cols = num_experts * hidden_dim
        # Checks
        if total_cols % tile_size != 0:
            raise ValueError(f"num_experts*hidden_dim={total_cols} not multiple of tile_size={tile_size}")
        if input_dim % tile_size != 0:
            raise ValueError(f"input_dim={input_dim} not multiple of tile_size={tile_size}")

        w_init = torch.randn(input_dim, total_cols)*0.01
        self.weight= nn.Parameter(w_init)

        # aggregator => (num_experts*hidden_dim -> output_dim)
        self.aggregator= nn.Linear(total_cols, output_dim)

    def forward(self, x):
        """
        x => (batch_size, in_dim)
        batch_size could be e.g. 64 => row_blocks=64/16=4 if tile_size=16
        leftover is dropped by dataloader => no partial leftover
        """
        batch_size= x.size(0)
        row_blocks= batch_size // self.tile_size  # e.g. 4

        # 1) gating => shape (batch_size, num_experts)
        gating_logits= self.gating_net(x) # [bs, K]
        # 2) sum across entire batch => shape (K,)
        gating_sum= gating_logits.sum(dim=0)
        # 3) pick top_k
        topk_vals, topk_inds= torch.topk(gating_sum, self.top_k)
        # build a length-K mask => 1 => topK experts
        block_mask_experts= torch.zeros(self.num_experts, dtype=torch.int32, device=x.device)
        block_mask_experts[topk_inds]=1

        # Because each expert is hidden_dim columns. If hidden_dim=16 => 1 block per expert horizontally.
        # => total block_cols= num_experts
        # replicate for row_blocks => final length= row_blocks * num_experts
        col_blocks= self.num_experts
        full_mask= block_mask_experts.unsqueeze(0).expand(row_blocks, col_blocks).reshape(-1)

        # 4) block-sparse => shape (batch_size, num_experts*hidden_dim)
        out_sparse= block_sparse_matmul(x, self.weight, full_mask, self.tile_size)

        # 5) aggregator => shape (batch_size, out_dim)
        final_out= self.aggregator(out_sparse)
        return final_out, None

# ======================================
# 4) Model Factory
# ======================================
def get_model(gating_type, num_experts, input_dim, hidden_dim, output_dim):
    gating_type= gating_type.lower()
    if gating_type=="block_sparse_moe":
        # top_k=3 for demonstration
        return DynamicBlockSparseMoE(input_dim, hidden_dim, output_dim, num_experts=num_experts,
                                     top_k=3, tile_size=16)
    else:
        raise ValueError(f"Unknown gating {gating_type}")

# ======================================
# 5) Evaluate
# ======================================
def evaluate_model(model, loader):
    mse= nn.MSELoss()
    model.eval()
    total_loss= 0.0
    total_samples= 0

    with torch.no_grad():
        for bx,by in loader:
            bx, by= bx.to(device), by.to(device)
            pred,_= model(bx)
            loss= mse(pred.squeeze(), by)
            bsz= bx.size(0)
            total_loss += loss.item()*bsz
            total_samples+= bsz

    model.train()
    if total_samples>0:
        return total_loss/ total_samples
    else:
        return 999.9

# ======================================
# 6) Train
# ======================================
def train_and_validate(model, train_loader, val_loader):
    optimizer= optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.5)
    mse= nn.MSELoss()

    best_val= float('inf')
    best_state= None
    start= time.time()
    model.train()

    for ep in tqdm(range(EPOCHS), desc="Training Epochs"):
        ep_loss= 0.0
        ep_samps= 0
        for bx,by in train_loader:
            bx, by= bx.to(device), by.to(device)

            optimizer.zero_grad()
            pred,_= model(bx)
            loss= mse(pred.squeeze(), by)
            loss.backward()
            optimizer.step()

            bsz= bx.size(0)
            ep_loss += loss.item()*bsz
            ep_samps+= bsz

        scheduler.step()
        if ep_samps>0:
            ep_loss /= ep_samps

        # Validation
        val_loss= evaluate_model(model,val_loader)
        if val_loss< best_val:
            best_val= val_loss
            best_state= {k:v.clone() for k,v in model.state_dict().items()}

        print(f"Epoch {ep+1}: TrainLoss={ep_loss:.4f}, ValLoss={val_loss:.4f}")

    train_time= time.time()-start
    if best_state is not None:
        model.load_state_dict(best_state)
    return train_time

# ======================================
# 7) Bootstrap
# ======================================
def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=95):
    means=[]
    n= len(data)
    for _ in range(n_bootstrap):
        idx= np.random.choice(n,n, replace=True)
        means.append(np.mean(data[idx]))
    means= np.array(means)
    lower= np.percentile(means, (100-ci)/2)
    upper= np.percentile(means, 100-(100-ci)/2)
    return np.mean(data), (lower, upper)

# ======================================
# 8) Main Experiment
# ======================================
def run_experiments():
    gating_types= SELECTED_GATING_TYPES
    results= {}
    test_results= {}
    test_results_raw= {}
    training_times= {}

    for num_exp in EXPERT_COUNTS:
        for gating_type in gating_types:
            print(f"\n=== Running gating='{gating_type}', num_experts={num_exp} ===")
            test_list= []
            total_time= 0.0

            for seed in range(N_RUNS):
                train_loader,val_loader,test_loader= generate_train_val_test_dataset(seed)

                base_model= get_model(gating_type,num_exp,INPUT_DIM,HIDDEN_DIM,OUTPUT_DIM).to(device)
                if MULTI_GPU:
                    model= nn.DataParallel(base_model)
                else:
                    model= base_model

                train_time= train_and_validate(model, train_loader, val_loader)
                test_mse= evaluate_model(model, test_loader)

                test_list.append(test_mse)
                total_time+= train_time

            training_times[(num_exp, gating_type)] = total_time/ N_RUNS
            test_results[(num_exp, gating_type)]    = np.mean(test_list)
            test_results_raw[(num_exp, gating_type)] = np.array(test_list)

    return training_times, test_results, test_results_raw

def plot_boxplots(test_results_raw):
    import matplotlib.pyplot as plt

    data=[]
    box_labels=[]
    for key, arr in test_results_raw.items():
        data.append(arr)
        box_labels.append(str(key))

    plt.figure(figsize=(8,6))
    bp= plt.boxplot(data, patch_artist=True, labels=box_labels)
    plt.ylabel("Test MSE")
    plt.title("Test MSE Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "test_mse_boxplots.png"))
    plt.close()

def main():
    training_times, test_results, test_results_raw= run_experiments()

    # Print summary
    print("\n=== Summary ===")
    for key in test_results.keys():
        mean_test= test_results[key]
        arr= test_results_raw[key]
        meanv, ci= bootstrap_confidence_interval(arr,1000,95)
        print(f"{key}: TestMSE={mean_test:.4f}, CI=({ci[0]:.4f},{ci[1]:.4f}), Time={training_times[key]:.2f}s")

    plot_boxplots(test_results_raw)
    print("\nDone! See results/plots in", SAVE_FOLDER)

if __name__=="__main__":
    main()
