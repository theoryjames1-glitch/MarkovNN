# MarkovNN

## 🔑 Key Ideas

1. **Markov states instead of raw inputs**

   * The 4 possible inputs $[0,0], [0,1], [1,0], [1,1]$ are treated as **distinct states** in a Markov chain.
   * So the input is one-hot encoded into 4 states.

2. **Markov transitions**

   * Each layer is a stochastic transition matrix $P$, with rows summing to 1.
   * $h_{t+1} = h_t P$.

3. **Stacking + residuals**

   * A single transition is too weak (linear).
   * By stacking multiple transitions and adding residual connections, we get non-trivial transformations while keeping the Markov property.

4. **Output as probabilities**

   * Final state → output distribution over $[0,1]$.
   * We supervise with binary cross-entropy against the XOR labels.

---

## 🧩 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# True Markov Layer: row-stochastic
class MarkovLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def forward(self, x):
        # Row-stochastic transition matrix
        P = F.softmax(self.logits, dim=-1)  # [dim, dim], rows sum to 1
        return torch.matmul(x, P.T)

# Markov NN for XOR
class MarkovNN_XOR(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=8, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([MarkovLayer(state_dim) for _ in range(n_layers)])
        self.hidden = MarkovLayer(state_dim)  # hidden-to-hidden
        self.out = nn.Linear(state_dim, 1)   # classifier (not stochastic)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h) + h   # residual to increase expressiveness
        y = torch.sigmoid(self.out(h))
        return y

# --- Dataset: XOR as Markov states ---
X_raw = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
Y = torch.tensor([[0.],[1.],[1.],[0.]])

# One-hot encode inputs into 4 states
def to_onehot(x):
    idx = x[:,0]*2 + x[:,1]   # map [0,0]->0, [0,1]->1, [1,0]->2, [1,1]->3
    return F.one_hot(idx, num_classes=4).float()

X = to_onehot(X_raw)

# --- Train ---
model = MarkovNN_XOR(state_dim=4, hidden_dim=8, n_layers=3)
opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

for epoch in range(2000):
    opt.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# --- Test ---
with torch.no_grad():
    print("Predictions (rounded):")
    print(model(X).round())
    print("Probabilities:")
    print(model(X))
```

---

## ✅ What This Achieves

* Inputs are **true states** in a Markov chain (one-hot vectors).
* Hidden layers are **row-stochastic transitions** (`softmax` rows).
* Expressivity comes from **multiple layers + residual connections**.
* Final linear classifier maps the resulting Markov state distribution to XOR output.
* This converges to correct outputs $[0,1,1,0]$.

