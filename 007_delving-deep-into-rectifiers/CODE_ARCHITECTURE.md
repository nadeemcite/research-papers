# Code Architecture — He Initialization from Scratch

## Notebook outline

| Section | Purpose |
|---|---|
| 1. Setup | Imports only NumPy, Matplotlib, and scikit-learn helpers |
| 2. Activations | ReLU and a learnable PReLU layer |
| 3. Deep MLP | Fully-connected network with selectable He / Xavier init |
| 4. Toy dataset | `make_moons` 2-D classification task |
| 5. Training helpers | Softmax, cross-entropy, SGD, and signal-flow diagnostics |
| 6. Experiment | Train two 10-layer ReLU nets: He vs Xavier |
| 7. Loss/accuracy plots | Side-by-side convergence |
| 8. Activation variance | Per-layer variance over training |
| 9. Gradient norms | Per-layer gradient magnitude over training |
| 10. Deeper stress test | 10 vs 20 layers to show Xavier stalls/degrades |
| 11. PReLU optional | Learnable slope with He-style factor |
| 12. Summary | Recap of formulas |

## Key functions and classes

### `relu(z)` / `relu_derivative(z)`
Simple element-wise `max(0, z)` and its binary derivative.

### `PReLU`
* `forward`: `z > 0 ? z : a * z`, stores `z` for backward.
* `backward`: returns gradient w.r.t. inputs and accumulates `grad_a`.
* `update`: SGD + momentum on the learned slope; no weight decay.

### `DeepMLP`
* Constructor builds a list of `(W, b, optional PReLU)` for each layer.
* `init_mode='he'` samples `W ~ N(0, 2/((1+a^2)*fan_in))`.
* `init_mode='xavier'` samples `W ~ N(0, 1/fan_in)`.
* `forward` stores activations and pre-activations for back-prop.
* `backward` computes gradients with chain rule through ReLU or PReLU.
* `update` applies SGD.

### `train(...)`
Mini-batch SGD loop. Every `snapshot_every` epochs it records:
* `act_var`: variance of each hidden activation tensor.
* `grad_norm`: Frobenius norm of each layer's weight gradient.

### `signal_flow_snapshot(...)`
A single forward/backward pass on a small fixed batch to produce the diagnostic numbers used by the plots.

## Data flow and shapes

```
Input X:            (B, 2)
Layer 1:            W1 (2, 64), b1 (64,)  -> z1 (B, 64) -> ReLU/PReLU -> a1 (B, 64)
...
Layer L (output):   WL (64, 2), bL (2,)   -> logits (B, 2) -> softmax (B, 2)
Loss:               cross_entropy(softmax(logits), Y_one_hot)
Backward:           dL/dlogits (B, 2) -> dW_i (fan_in, fan_out), db_i (fan_out,)
```

## Deliberate simplifications vs. full paper

1. **Dataset:** We use a 2-D toy problem instead of ImageNet. This keeps runtime under a minute while still demonstrating vanishing/exploding signal behavior.
2. **Architecture:** We implement fully-connected MLPs rather than the paper's convolutional SPP-net variants. He initialization applies to both; the math is identical for `fan_in`.
3. **No batch normalization:** The paper predates BN and aims to show that careful init alone can train deep nets. We stick to that spirit.
4. **PReLU is optional:** The main diagnostic experiment uses plain ReLU so the contrast between He and Xavier is clearest; PReLU is exercised in a short follow-up cell.
5. **SGD only:** We omit momentum/weight-decay schedules so the focus stays on initialization. The formulas are documented in the README.
