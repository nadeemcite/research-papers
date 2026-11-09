---
title: "Distillation: Code architecture"
parent: "045 Distillation"
nav_order: 1
---
# Code Architecture — Knowledge Distillation Notebook

## Overview

The notebook implements knowledge distillation from Hinton et al. (2015) from scratch in PyTorch. A large teacher MLP is trained on MNIST, then a smaller student MLP is trained three ways: (1) from hard labels only, (2) from the teacher's soft targets via distillation, and (3) from a weighted combination of soft and hard targets. The three students are compared on test accuracy, and the teacher's soft-target distributions are visualized to show the "dark knowledge" being transferred. All computation is CPU-friendly (small MLPs on MNIST, no GPU required).

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `torch.nn`, `torch.nn.functional`, `torchvision` for MNIST
- `numpy`, `matplotlib` for visualization
- Random seed fixing for reproducibility
- Device set to CPU (this is a CPU-friendly notebook)

### 2. Data Loading — MNIST
- Load MNIST train/test sets via `torchvision.datasets.MNIST`
- Normalize with the MNIST mean and standard deviation
- DataLoaders with batch_size=128 for training, 256 for testing
- Data shapes: train images (60000, 1, 28, 28), train labels (60000,)

### 3. Teacher Network (Large MLP)
- **Class `TeacherNet(nn.Module)`**
  - Input: 784 (flattened 28×28)
  - Hidden: two layers of 800 ReLU units
  - Dropout (0.5) on hidden layers
  - Output: 10 logits (MNIST digits)
  - Forward: 784 → 800 → Dropout → 800 → Dropout → 10
  - 1,276,810 parameters

### 4. Student Network (Small MLP)
- **Class `StudentNet(nn.Module)`**
  - Input: 784
  - Hidden: two layers of 300 ReLU units
  - No dropout
  - Output: 10 logits
  - Forward: 784 → 300 → 300 → 10
  - 328,810 parameters (3.9x fewer than the teacher's 1,276,810)

### 5. Softmax with Temperature — From Scratch
- **Function `softmax_with_temperature(logits, T)`**
  - Input: logits (N, 10), temperature T (scalar)
  - Compute: exp(logits / T) / sum(exp(logits / T)), after subtracting the maximum for numerical stability
  - Returns soft probability distribution (N, 10)
  - The demo prints the output for temperatures of 1, 2, 5 and 10; the main runs use 4

### 6. Distillation Loss — From Scratch
- **Function `distillation_loss(student_logits, teacher_logits, T)`**
  - Uses `F.softmax` and `F.log_softmax` at temperature T for the teacher and student
  - Loss: `F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')`, which is KL(teacher \|\| student)
  - Multiply by T^2 to compensate for gradient scaling (as specified in the paper)
  - Key shapes: student_logits (B, 10), teacher_logits (B, 10) → loss (scalar)

### 7. Combined Loss Function
- **Function `combined_loss(student_logits, teacher_logits, labels, T, alpha)`**
  - Soft loss: alpha * T^2 * KL(teacher_soft \|\| student_soft)
  - Hard loss: (1 - alpha) * CrossEntropy(student_logits, labels)
  - alpha controls the trade-off; this notebook uses 0.7
  - Returns total loss

### 8. Training the Teacher
- Train TeacherNet on MNIST for 10 epochs
- Optimizer: Adam, lr=0.001
- Loss: standard CrossEntropy with hard labels
- Record training loss, test accuracy per epoch
- Saved run: 98.00% test accuracy

### 9. Training Students — Three Conditions
- **Student-Hard**: trained with standard CrossEntropy on hard labels only
- **Student-Distill**: trained with distillation loss only (soft targets, T=4)
- **Student-Combined**: trained with combined loss (alpha=0.7, T=4)
- All three use identical architecture (StudentNet), same data, same optimizer (Adam, lr=0.001)
- Train each for 10 epochs
- Record training loss, test accuracy per epoch for comparison

### 10. Dark Knowledge Visualization
- For a sample of test images, extract the teacher's soft predictions at temperatures of 1 and 10
- Show that the teacher assigns small but non-trivial probabilities to "wrong" classes
- These ratios carry the similarity structure ("which 2s look like 3s")
- Visualize as heatmaps: teacher soft probs for several test images

### 11. Temperature Sensitivity Experiment
- Train separate distilled students at T = [1, 2, 4, 8, 16, 20]
- Plot final test accuracy vs. temperature
- Saved run: best T=4 at 98.01%, worst T=2 at 97.67%, with no clear trend

### 12. Comparison of Soft vs Hard Target Gradients
- For a single training example, compute the gradient magnitude from:
  - Hard label loss (one-hot target)
  - Soft target loss (teacher's softened distribution at T=4)
- Saved run: soft targets had a higher coefficient of variation (0.192) than hard targets (0.155), on one batch with an untrained student

### 13. Visualization & Results
- Plot 1: Training loss curves (teacher, student-hard, student-distill, student-combined)
- Plot 2: Test accuracy curves over epochs for all three student conditions
- Plot 3: Teacher soft target distribution for sample images (bar charts at T=1 vs T=10)
- Plot 4: Temperature sensitivity — accuracy vs. temperature
- Plot 5: Confusion matrices for student-hard vs student-distill

### 14. Summary & Analysis
- Print final test accuracies for all models
- Saved run: combined 97.92%, hard labels 97.85%, distill only 97.78%, teacher 98.00%; the gaps are within the noise of this setup
- Compare parameter counts: teacher vs student

## Key Functions/Classes
| Component | Purpose |
|---|---|
| `TeacherNet` | Large MLP (800-800-10) with dropout |
| `StudentNet` | Small MLP (300-300-10) without dropout |
| `softmax_with_temperature()` | Temperature-scaled softmax from scratch |
| `distillation_loss()` | KL-divergence-based soft target loss |
| `combined_loss()` | Weighted combination of soft + hard losses |
| `train_model()` | Generic training loop supporting any loss function |
| `evaluate()` | Test accuracy evaluation |

## Data Flow & Shapes
```
MNIST image: (1, 28, 28) → flatten → (784,)
  ↓ TeacherNet
  teacher_logits: (10,) → softmax(T=4) → teacher_soft: (10,)
  ↓ StudentNet
  student_logits: (10,) → softmax(T=4) → student_soft: (10,)
  ↓
  distillation_loss = T^2 * KL(teacher_soft, student_soft)
  hard_loss = CrossEntropy(student_logits, true_label)
  total = alpha * distillation_loss + (1-alpha) * hard_loss
```

## Deliberate Simplifications vs Full Paper
1. **MNIST only** — the paper also reports speech recognition and JFT image experiments
2. **Single teacher** — the paper's speech experiment distils an ensemble of 10 models; its MNIST experiment and this notebook use a single large net
3. **Small MLPs** — the paper's MNIST nets use 1200 and 800 units per layer; we use 800 and 300 for faster CPU training
4. **No specialist models** — paper introduced ensembles of specialists for JFT; we focus on the core distillation mechanism
5. **No transfer-set experiments** — paper showed distillation works even when transfer set omits entire classes; we use the full training set
6. **Fixed alpha** — we use alpha=0.7 throughout
7. **No speech/ASR experiments** — the Android voice search experiments require proprietary data and massive compute
