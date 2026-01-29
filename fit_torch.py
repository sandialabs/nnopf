import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
import random
from IPython import embed


case_name = "pglib_opf_case30_ieee"

# -------------------------
# Dataset
# -------------------------
train_ds_full = OPFDataset(root="data", case_name=case_name, split="train")
val_ds_full = OPFDataset(root="data", case_name=case_name, split="val")

test_ds_full = OPFDataset(root="data", case_name=case_name, split="test")

i = len(val_ds_full) // 100
indices = random.sample(range(len(val_ds_full)), i)
val_ds = torch.utils.data.Subset(val_ds_full, indices)

j = len(test_ds_full) // 100
indices = random.sample(range(len(test_ds_full)), j)
test_ds = torch.utils.data.Subset(test_ds_full, indices)


k = len(train_ds_full) // 100
indices = random.sample(range(len(train_ds_full)), k)

train_ds = torch.utils.data.Subset(train_ds_full, indices)

training_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


# -------------------------
# Helper: flatten inputs
# -------------------------
def flatten_inputs(data):
    """
    Converts hetero node features into a single vector.
    Assumes fixed topology (IEEE-14).
    """
    xs = []

    for node_type in ["load"]:
        x = data.x_dict[node_type]
        xs.append(x.flatten())

    return torch.cat(xs, dim=0)


def flatten_targets(data):
    """
    Generator active/reactive power outputs
    """
    ys = []

    for node_type in ["generator", "bus"]:
        y = data.y_dict[node_type]
        ys.append(y.flatten())

    return torch.cat(ys, dim=0)


# Infer input/output sizes
sample = train_ds[0]
input_dim = flatten_inputs(sample).numel()
output_dim = flatten_targets(sample).numel()

print(f"Input dim: {input_dim}")
print(f"Output dim: {output_dim}")


# -------------------------
# Feed-Forward Model
# -------------------------
class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim),
        )

    def forward(self, x):
        return self.net(x)


model = FFNN(input_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0

    for data in loader:
        x = flatten_inputs(data)
        y = flatten_targets(data)

        pred = model(x)
        loss = F.mse_loss(pred, y)

        total_loss += loss.item()

    model.train()
    return total_loss / len(loader)


# -------------------------
# Training loop
# -------------------------

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_state_dict = None

model.train()
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = 0.0

    for data in training_loader:
        optimizer.zero_grad()

        x = flatten_inputs(data)
        y = flatten_targets(data)

        pred = model(x)
        loss = F.mse_loss(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(training_loader)
    val_loss = evaluate_loss(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # --- checkpoint best model ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:03d} | " f"Train: {train_loss:.6f} | " f"Val: {val_loss:.6f}")


@torch.no_grad()
def compute_r2(model, loader):
    model.eval()

    preds = []
    targets = []

    for data in loader:
        x = flatten_inputs(data)
        y = flatten_targets(data)

        pred = model(x)

        preds.append(pred)
        targets.append(y)

    preds = torch.stack(preds)  # [N, output_dim]
    targets = torch.stack(targets)  # [N, output_dim]

    # R² per output dimension
    ss_res = ((targets - preds) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)

    r2 = 1 - ss_res / ss_tot

    return r2


@torch.no_grad()
def compute_mse_per_output(model, loader):
    model.eval()

    preds = []
    targets = []

    for data in loader:
        x = flatten_inputs(data)
        y = flatten_targets(data)

        pred = model(x)

        preds.append(pred)
        targets.append(y)

    preds = torch.stack(preds)  # [N, output_dim]
    targets = torch.stack(targets)  # [N, output_dim]

    mse_per_output = ((preds - targets) ** 2).mean(dim=0)

    return mse_per_output


model.load_state_dict(best_state_dict)
print(f"\nRestored best model (val MSE = {best_val_loss:.6f})")

test_loss = evaluate_loss(model, test_loader)
r2 = compute_r2(model, test_loader)

print(f"\nTest MSE: {test_loss:.6f}")
print("R² per output:")

for i, r in enumerate(r2):
    print(f" Output {i:02d}: R² = {r.item():.4f}")

mse_per_output = compute_mse_per_output(model, test_loader)

pg_MSE = []
qg_MSE = []
va_MSE = []
vm_MSE = []
print("\nMSE per output:")
for i, mse in enumerate(mse_per_output):
    print(f" Output {i:02d}: MSE = {mse.item():.6e}")
    if i < 12 and i % 2 == 0:
        pg_MSE.append(mse.item())
    elif i < 12 and i % 2 != 0:
        qg_MSE.append(mse.item())
    elif i >= 12 and i % 2 == 0:
        va_MSE.append(mse.item())
    elif i >= 12 and i % 2 != 0:
        vm_MSE.append(mse.item())


# save_path = f"trained_models/ffnn_{case_name}_best.pt"
# torch.save(model.state_dict(), save_path)
print(val_losses)
print(train_losses)


# import plotly.graph_objects as go

# Create a figure
# fig = go.Figure()

# Add traces for training and validation losses
# fig.add_trace(go.Scatter(x=list(range(len(train_losses))), y=train_losses, mode='lines', name='Train MSE'))
# fig.add_trace(go.Scatter(x=list(range(len(val_losses))), y=val_losses, mode='lines', name='Val MSE'))

# Update layout
# fig.update_layout(
#    title=f'Training Curve ({case_name})',
#    xaxis_title='Epoch',
#    yaxis_title='MSE',
#    legend_title='Legend',
#    template='plotly_white'  # Optional: Use a white background
# )

# Show grid lines
# fig.update_xaxes(showgrid=True)
# fig.update_yaxes(showgrid=True)

# Save the figure as a PNG file
# fig.write_image('TRAINFIG.png')

# Show the figure
# fig.show()

embed()
