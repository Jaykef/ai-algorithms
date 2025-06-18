import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import math
from tqdm import trange
import matplotlib.pyplot as plt
from collections import Counter

# ==========================================================
# Self-Compressing Neural Networks
# Here we implement a minimal version of the paper "Self-compressing 
# neural networks" It shows dynamic neural network compression during training - 
# reduced size of weight, activation tensors and bits required to represent weight
# Paper: https://arxiv.org/pdf/2301.13142
# Demo: https://x.com/Jaykef_/status/1821518359122280482
# ==========================================================

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Split data
X_train, Y_train = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))))
X_test, Y_test = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))

class QConv2d(nn.Module):
    """
    Quantized Conv2d layer starting with 2 bits per weight
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(QConv2d, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        scale = 1 / math.sqrt(in_channels * math.prod(self.kernel_size))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size).uniform_(-scale, scale))
        self.e = nn.Parameter(torch.full((out_channels, 1, 1, 1), -8.))
        self.b = nn.Parameter(torch.full((out_channels, 1, 1, 1), 2.))  # start with 2 bits per weight

    def qbits(self):
        return self.b.relu().sum() * math.prod(self.weight.shape[1:])

    def qweight(self):
        return torch.minimum(torch.maximum(2**-self.e * self.weight, -2**(self.b.relu()-1)), 2**(self.b.relu()-1) - 1)

    def forward(self, x):
        qw = self.qweight()
        w = (qw.round() - qw).detach() + qw  # straight through estimator
        return nn.functional.conv2d(x, 2**self.e * w)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            QConv2d(1, 32, 5), nn.ReLU(),
            QConv2d(32, 32, 5), nn.ReLU(),
            nn.BatchNorm2d(32, affine=False, track_running_stats=False),
            nn.MaxPool2d(2),
            QConv2d(32, 64, 3), nn.ReLU(),
            QConv2d(64, 64, 3), nn.ReLU(),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(576, 10)  # 576 = 64 * 3 * 3

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

    def qbits(self):
        return sum(l.qbits() for l in self.features if isinstance(l, QConv2d))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
opt = optim.Adam(model.parameters())
test_accs, bytes_used = [], []
weight_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_step():
    model.train()
    opt.zero_grad()
    samples = torch.randint(0, X_train.shape[0], (512,))
    outputs = model(X_train[samples].to(device))
    loss = nn.functional.cross_entropy(outputs, Y_train[samples].to(device))
    Q = model.qbits() / weight_count
    loss = loss + 0.05 * Q  # hyperparameter determines compression vs accuracy
    loss.backward()
    opt.step()
    return loss.item(), Q.item()

def get_test_acc():
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        pred = outputs.argmax(dim=1)
        return (pred == Y_test.to(device)).float().mean().item() * 100

test_acc = 0.0
test_accs, bytes_used = [], []
# Training loop
for i in (t := trange(4000)):
    loss, Q = train_step()
    model_bytes = Q / 8 * weight_count
    if i % 10 == 9:
        test_acc = get_test_acc()
    test_accs.append(test_acc)
    bytes_used.append(model_bytes)
    t.set_description(f"loss: {loss:6.2f}  bytes: {model_bytes:.1f}  acc: {test_acc:5.2f}%")


fig, ax1 = plt.subplots(figsize=(12,6))

# Plot model size
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Model Size (bytes)', color='red')
ax1.plot(bytes_used, color='red')
ax1.tick_params(axis='y')

# Plot test accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Test Accuracy (%)', color='blue')
ax2.plot(test_accs, color='blue')
ax2.tick_params(axis='y')

ax2.set_ylim(80, 100)
plt.title('Model Size vs Test Accuracy')
plt.show()
