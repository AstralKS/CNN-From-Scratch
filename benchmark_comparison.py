"""
Benchmark: CNN from Scratch (NumPy/SciPy) vs PyTorch
Trains both on MNIST binary classification (0 vs 1) and generates comparison graphs.
"""
# ========================== THIS IS AI GENERATED CODE FOR CLARIFICATION ==========================

import numpy as np
import time
import os

np.random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('ggplot')

from torchvision import datasets

# Scratch CNN imports
from dense import Dense
from activations import Sigmoid
from convolutional import Convolutional
from reshape import Reshape
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import predict

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# ========================== Configuration ==========================
SAMPLES_PER_CLASS = 100
EPOCHS = 20
LEARNING_RATE = 0.1
KERNEL_SIZE = 3
CONV_DEPTH = 5
CONV_OUT = 26  # 28 - KERNEL_SIZE + 1
HIDDEN = 100
NUM_CLASSES = 2
OUTPUT_DIR = "graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

# ========================== Data ==========================
def preprocess_data(images, labels, limit):
    zero_idx = np.where(labels == 0)[0][:limit]
    one_idx  = np.where(labels == 1)[0][:limit]
    idx = np.random.permutation(np.hstack((zero_idx, one_idx)))
    x, y = images[idx], labels[idx]
    x = x.reshape(len(x), 1, 28, 28).astype("float32") / 255
    y_cat = to_categorical(y, NUM_CLASSES)
    y_cat = y_cat.reshape(len(y_cat), NUM_CLASSES, 1)
    return x, y_cat

print("Loading MNIST...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True)
mnist_test  = datasets.MNIST(root='./data', train=False, download=True)
x_tr_raw = mnist_train.data.numpy()
y_tr_raw = mnist_train.targets.numpy()
x_te_raw = mnist_test.data.numpy()
y_te_raw = mnist_test.targets.numpy()
x_train, y_train = preprocess_data(x_tr_raw, y_tr_raw, SAMPLES_PER_CLASS)
x_test,  y_test  = preprocess_data(x_te_raw, y_te_raw, SAMPLES_PER_CLASS)
print(f"Train: {len(x_train)} samples  |  Test: {len(x_test)} samples")

FLAT = CONV_DEPTH * CONV_OUT * CONV_OUT

# ========================== 1. Scratch CNN ==========================
print("\n" + "=" * 60)
print("  Training CNN from Scratch (NumPy + SciPy)")
print("=" * 60)

network = [
    Convolutional((1, 28, 28), KERNEL_SIZE, CONV_DEPTH),
    Sigmoid(),
    Reshape((CONV_DEPTH, CONV_OUT, CONV_OUT), (FLAT, 1)),
    Dense(FLAT, HIDDEN),
    Sigmoid(),
    Dense(HIDDEN, NUM_CLASSES),
    Sigmoid()
]

scratch_losses = []
scratch_accs   = []
scratch_times  = []

for ep in range(EPOCHS):
    t0 = time.time()
    err = 0
    for x, y in zip(x_train, y_train):
        out = predict(network, x)
        err += binary_cross_entropy(y, out)
        grad = binary_cross_entropy_prime(y, out)
        for layer in reversed(network):
            grad = layer.backward(grad, LEARNING_RATE)
    err /= len(x_train)
    dt = time.time() - t0

    correct = sum(1 for x, y in zip(x_test, y_test)
                  if np.argmax(predict(network, x)) == np.argmax(y))
    acc = correct / len(x_test) * 100

    scratch_losses.append(float(err))
    scratch_accs.append(acc)
    scratch_times.append(dt)
    print(f"  Epoch {ep+1:2d}/{EPOCHS}  |  Loss: {err:.4f}  |  Acc: {acc:5.1f}%  |  {dt:.2f}s")

scratch_total = sum(scratch_times)
print(f"  Total: {scratch_total:.1f}s")
scratch_filters = network[0].kernals.copy()

# ========================== 2. PyTorch CNN ==========================
print("\n" + "=" * 60)
print("  Training PyTorch CNN (same architecture)")
print("=" * 60)

torch.manual_seed(42)

class PtCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, CONV_DEPTH, kernel_size=KERNEL_SIZE),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(FLAT, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, NUM_CLASSES),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = PtCNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

x_tr_pt = torch.FloatTensor(x_train)
y_tr_pt = torch.FloatTensor(y_train.reshape(-1, NUM_CLASSES))
x_te_pt = torch.FloatTensor(x_test)
y_te_labels = np.argmax(y_test.reshape(-1, NUM_CLASSES), axis=1)

pt_losses = []
pt_accs   = []
pt_times  = []

for ep in range(EPOCHS):
    t0 = time.time()
    model.train()
    epoch_loss = 0
    for i in range(len(x_tr_pt)):
        optimizer.zero_grad()
        out = model(x_tr_pt[i:i+1])
        loss = criterion(out, y_tr_pt[i:i+1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(x_tr_pt)
    dt = time.time() - t0

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(x_te_pt), dim=1).numpy()
        acc = (preds == y_te_labels).mean() * 100

    pt_losses.append(epoch_loss)
    pt_accs.append(acc)
    pt_times.append(dt)
    print(f"  Epoch {ep+1:2d}/{EPOCHS}  |  Loss: {epoch_loss:.4f}  |  Acc: {acc:5.1f}%  |  {dt:.4f}s")

pt_total = sum(pt_times)
print(f"  Total: {pt_total:.2f}s")
pt_filters = model.net[0].weight.detach().numpy()

# ========================== Graphs ==========================
print("\n" + "=" * 60)
print("  Generating Graphs")
print("=" * 60)

epochs_x = range(1, EPOCHS + 1)
C_SCRATCH = '#E74C3C'
C_PYTORCH = '#3498DB'
C_GREEN   = '#27AE60'
BG        = '#FAFAFA'

# ---------- 1. Loss Comparison ----------
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(epochs_x, scratch_losses, 'o-', color=C_SCRATCH, lw=2.5, ms=6,
        label='CNN from Scratch (NumPy)')
ax.plot(epochs_x, pt_losses, 's-', color=C_PYTORCH, lw=2.5, ms=6,
        label='PyTorch CNN')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=13)
ax.set_title('Training Loss Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_comparison.png'), dpi=150)
plt.close()
print("  [saved] loss_comparison.png")

# ---------- 2. Accuracy Comparison ----------
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(epochs_x, scratch_accs, 'o-', color=C_SCRATCH, lw=2.5, ms=6,
        label='CNN from Scratch (NumPy)')
ax.plot(epochs_x, pt_accs, 's-', color=C_PYTORCH, lw=2.5, ms=6,
        label='PyTorch CNN')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Test Accuracy Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), dpi=150)
plt.close()
print("  [saved] accuracy_comparison.png")

# ---------- 3. Training Time Comparison ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)

ax = axes[0]; ax.set_facecolor(BG)
bars = ax.bar(['Scratch\n(NumPy/SciPy)', 'PyTorch'],
              [scratch_total, pt_total],
              color=[C_SCRATCH, C_PYTORCH], width=0.5, alpha=0.85,
              edgecolor='white', linewidth=2)
for b, v in zip(bars, [scratch_total, pt_total]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
            f'{v:.1f}s', ha='center', fontsize=13, fontweight='bold')
ax.set_ylabel('Seconds', fontsize=13)
ax.set_title('Total Training Time', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
speedup = scratch_total / pt_total if pt_total > 0 else float('inf')
ax.annotate(f'PyTorch is {speedup:.1f}x faster',
            xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', fc='#ECF0F1', ec='#BDC3C7'))

ax = axes[1]; ax.set_facecolor(BG)
ax.plot(epochs_x, scratch_times, 'o-', color=C_SCRATCH, lw=2, ms=5, label='Scratch')
ax.plot(epochs_x, pt_times, 's-', color=C_PYTORCH, lw=2, ms=5, label='PyTorch')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Seconds', fontsize=13)
ax.set_title('Per-Epoch Training Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'time_comparison.png'), dpi=150)
plt.close()
print("  [saved] time_comparison.png")

# ---------- 4. Sample Predictions ----------
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.patch.set_facecolor(BG)
fig.suptitle('Sample Predictions: Scratch vs PyTorch',
             fontsize=16, fontweight='bold', y=1.02)

model.eval()
np.random.seed(123)
sample_idx = np.random.choice(len(x_test), 5, replace=False)

for col, si in enumerate(sample_idx):
    img = x_test[si].squeeze()
    true_lbl = int(np.argmax(y_test[si]))

    s_out = predict(network, x_test[si])
    s_pred = int(np.argmax(s_out))
    s_conf = float(s_out.flatten()[s_pred])

    with torch.no_grad():
        p_out = model(x_te_pt[si:si+1])
    p_pred = int(torch.argmax(p_out).item())
    p_conf = float(p_out[0, p_pred].item())

    for row, (pred, conf, label) in enumerate([
        (s_pred, s_conf, 'Scratch'),
        (p_pred, p_conf, 'PyTorch')
    ]):
        ax = axes[row, col]
        ax.imshow(img, cmap='gray')
        clr = C_GREEN if pred == true_lbl else C_SCRATCH
        ax.set_title(f'Pred: {pred} ({conf:.0%})\nTrue: {true_lbl}',
                     fontsize=10, color=clr, fontweight='bold')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(label, fontsize=13, fontweight='bold', labelpad=10)
            ax.yaxis.set_visible(True); ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  [saved] sample_predictions.png")

# ---------- 5. Learned Convolutional Filters ----------
fig, axes = plt.subplots(2, CONV_DEPTH, figsize=(3 * CONV_DEPTH, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Learned Convolutional Filters (Scratch vs PyTorch)',
             fontsize=16, fontweight='bold')

for i in range(CONV_DEPTH):
    for row, (filt, label) in enumerate([
        (scratch_filters[i, 0], 'Scratch'),
        (pt_filters[i, 0], 'PyTorch')
    ]):
        ax = axes[row, i]
        ax.imshow(filt, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f'Filter {i+1}', fontsize=11)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel(label, fontsize=13, fontweight='bold', labelpad=10)
            ax.yaxis.set_visible(True); ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learned_filters.png'), dpi=150)
plt.close()
print("  [saved] learned_filters.png")

# ---------- 6. Combined Dashboard ----------
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG)
fig.suptitle('CNN from Scratch vs PyTorch — Full Comparison',
             fontsize=20, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor(BG)
ax1.plot(epochs_x, scratch_losses, 'o-', color=C_SCRATCH, lw=2, ms=4, label='Scratch')
ax1.plot(epochs_x, pt_losses, 's-', color=C_PYTORCH, lw=2, ms=4, label='PyTorch')
ax1.set_title('Training Loss', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor(BG)
ax2.plot(epochs_x, scratch_accs, 'o-', color=C_SCRATCH, lw=2, ms=4, label='Scratch')
ax2.plot(epochs_x, pt_accs, 's-', color=C_PYTORCH, lw=2, ms=4, label='PyTorch')
ax2.set_title('Test Accuracy', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 105); ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor(BG)
bars = ax3.bar(['Scratch', 'PyTorch'], [scratch_total, pt_total],
               color=[C_SCRATCH, C_PYTORCH], alpha=0.85, width=0.45)
for b, v in zip(bars, [scratch_total, pt_total]):
    ax3.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
             f'{v:.1f}s', ha='center', fontsize=11, fontweight='bold')
ax3.set_title('Total Training Time', fontweight='bold')
ax3.set_ylabel('Seconds'); ax3.grid(axis='y', alpha=0.3)

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor(BG)
final_accs = [scratch_accs[-1], pt_accs[-1]]
bars = ax4.bar(['Scratch', 'PyTorch'], final_accs,
               color=[C_SCRATCH, C_PYTORCH], alpha=0.85, width=0.45)
for b, v in zip(bars, final_accs):
    ax4.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
             f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax4.set_title('Final Test Accuracy', fontweight='bold')
ax4.set_ylabel('Accuracy (%)'); ax4.set_ylim(0, 110)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [saved] dashboard.png")

# ========================== Summary ==========================
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Scratch  — Loss: {scratch_losses[-1]:.4f}  Acc: {scratch_accs[-1]:.1f}%  Time: {scratch_total:.1f}s")
print(f"  PyTorch  — Loss: {pt_losses[-1]:.4f}  Acc: {pt_accs[-1]:.1f}%  Time: {pt_total:.2f}s")
print(f"  Speedup  — PyTorch is {speedup:.1f}x faster")
print(f"\n  6 graphs saved to '{OUTPUT_DIR}/' :")
print(f"    1. loss_comparison.png")
print(f"    2. accuracy_comparison.png")
print(f"    3. time_comparison.png")
print(f"    4. sample_predictions.png")
print(f"    5. learned_filters.png")
print(f"    6. dashboard.png  (all-in-one for blog header)")
print("=" * 60)
