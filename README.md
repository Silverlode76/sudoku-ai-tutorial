<img width="50%"  alt="image" align="right" src="https://github.com/user-attachments/assets/97b00694-5810-409a-8309-181d6328f322"/>

# Sudoku AI Tutorial

A hands-on educational project that demonstrates how a Sudoku puzzle can be solved using *tensors, logits, softmax, constraint loss functions,* and the *Adam optimizer* in PyTorch.

This tutorial teaches beginners the fundamentals of:
- tensor operations
- probability distributions via softmax
- row/column/block constraints
- optimizer-based reasoning
- temperature annealing
- implementing custom loss functions

The full example is built on a 4×4 Sudoku puzzle and extends naturally to 6×6 and 9×9 grids.

# Introduction
This course explains how a Sudoku puzzle can be solved with an optimizer approach. We use tensor operations to define the loss functions needed for the optimizer.

## The 4x4 Sudoku problem
A Sudoku puzzle is solved if every row contains the numbers 1,2,3,4 and every column contains the numbers 1,2,3,4. Additionally, each of the four blocks must also contain the numbers 1,2,3,4. Mathematically this is expressed as follows:

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/2ce3e79d-d94a-4239-b4df-99845c8727d7" />

## The 4x4 Sudoku Puzzle
Let’s assume we have a Sudoku puzzle where only four numbers are given. We can place all givens with their values in a matrix as shown below:

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/a6b2e530-1f56-4b17-a2b6-0bebd7972f64" />

## The 4x4 Sudoku Tensors
Since we created a matrix representing our puzzle with the given numbers, we need to define the matrix and the givens as tensors so we can manipulate them in PyTorch. Because tensor indices start with 0, we subtract 1 from puzzle values (which are between 1..4) so they map to tensor indices 0..3. The table below shows the Sudoku puzzle, the Sudoku tensor, and the logical givens:

| Step | Description | Image |
|------|-------------|--------|
| 1 | Input Sudoku | <img width="208" height="148" alt="image" src="https://github.com/user-attachments/assets/4006291d-f159-4fc7-ba38-6f30fbd381c5" />|
| 2 | Input Sudoku Tensor | <img width="200" height="130" alt="image" src="https://github.com/user-attachments/assets/634c4231-ad17-4bac-8806-b02410a738a5" />|
| 3 | Input Givens | <img width="218" height="130" alt="image" src="https://github.com/user-attachments/assets/04be267b-fa61-46b4-9479-b6e53dc30574" />|

# Preparing the Z Tensor (The Puzzle Tensor)
Since we have the Sudoku tensor and the givens, we can now create our Z tensor, which contains the digits for our loss functions. Next, we add the digit dimension so the tensor represents row, column, and value (digits 0..3). Our Z cube looks like this (red and green are givens from our puzzle). Each digit already set as a given is marked with 5.

| Tensor Z | Description |
|------|-------------|
|<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/c4441557-83a9-4209-b140-8f05d1fa4501" />|<img width="1186" height="666" alt="image" src="https://github.com/user-attachments/assets/2f62fa3a-2a6e-431b-bd5e-b36487ba09bd" />|

# Preparing the Optimizer
We use the Adam optimizer. The input is our tensor Z:
```python
opt = torch.optim.Adam([Z], lr=lr)
```
# The Adam Optimizer
Why we use the Adam optimizer
- We optimize a single trainable tensor Z, which contains the logits for each Sudoku cell.
- The optimization landscape of Sudoku is highly non-linear and contains many local minima due to:
  - row/column/block constraints
  - given-value constraints
  - temperature-scaled softmax
  - multiple interacting loss terms

Adam is ideal here because:
- it adapts the learning rate per tensor element
- it combines Momentum and RMSProp, providing stability
- it handles flat regions and steep regions effectively
- it converges much faster than SGD
- it works very well with softmax and temperature annealing

## Temperature Annealing / Softmax

P is the probability density of tensor Z.
Example of Softmax():

<img width="364" height="148" alt="image" src="https://github.com/user-attachments/assets/408b6f5c-d844-4803-a8e6-71258cb1d2e8" />

Why Softmax
Softmax ensures that each cell represents a soft decision over all possible digits.
Each cell has a “soft” decision over the possible digits (1–4).
P[r, c, :] is always a valid probability vector — all entries are positive.
The probabilities always sum to 1, making them perfect for enforcing Sudoku constraints.
The optimizer can learn smoothly because these soft probabilities are differentiable, unlike hard one-hot choices

```python
t = temperature_init + (temperature_final - temperature_init) * (step / max_steps)
P = F.softmax(Z / t, dim=2)  # (4,4,4)
```
The tensors Z and P have the dimensions row (dim=0), col (dim=1), and digit (dim=2). In the function above we do softmax over the digits, therefore we set dim=2. This means that if we sum up, for every element such as P(0,0,:), we get a sum of 1 as shown below:

| Tensor Z | Tensor P |
|------|-------------|
| <img width="80%" alt="image" src="https://github.com/user-attachments/assets/18fbd8c2-362c-4191-b40a-44600da8ff15" /> | <img width="80%" alt="image" src="https://github.com/user-attachments/assets/88e32fa6-169f-4b44-ae4a-f6e63770ad17" />|

Let's consider slice = 0 (row = 0). The sum over the digits is always 1.0.

| Z-Vector (Logits)                            | P-Vector (Softmax)                 | Sum(P) |
| -------------------------------------------- | ---------------------------------- | ------ |
| `[5.0, -5.0, -5.0, -5.0]`                    | `[0.9424, 0.0202, 0.0202, 0.0172]` | `1.0`  |
| `[-0.0059, -0.0025, 0.0057, -0.0039]`        | `[0.2296, 0.2698, 0.2707, 0.2298]` | `1.0`  |
| `[4.7337e-04, -2.0038e-03, 7.8445e-03, ...]` | `[0.2298, 0.2694, 0.2705, 0.2303]` | `1.0`  |
| `[-5.0, -5.0, -5.0, 5.0]`                    | `[0.0172, 0.0202, 0.0202, 0.9424]` | `1.0`  |

# The Loss Function

## The Row Loss
For each row we determine the loss with:

<img width="30%" alt="image" src="https://github.com/user-attachments/assets/76bff619-6583-46b4-bdea-beb337c8de4a" />

```python
row_sum = P.sum(dim=1)          # (4,4) over columns -> rows x digits
L_row = ((row_sum - 1.0) ** 2).sum()
```
The tensor `row_sum` has the dimensions (row, digits) after the operation above. For demonstration, take the first slice = 0 (row 0). Each column provides a probability distribution over the four digits (column sums are 1). `row_sum[0]` adds these columns per digit and yields a two-dimensional table (row × digit):

| Digit | P[0,0,d] | P[0,1,d] | P[0,2,d] | P[0,3,d] | row_sum[0, d] = sum over columns |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.70 | 0.05 | 0.20 | 0.05 | **1.00** |
| 1 | 0.10 | 0.05 | 0.30 | 0.35 | **0.80** |
| 2 | 0.10 | 0.80 | 0.10 | 0.10 | **1.10** |
| 3 | 0.10 | 0.10 | 0.40 | 0.50 | **1.10** |

`row_sum[0] = [1.00, 0.80, 1.10, 1.10]`. Ideally each digit should be `1.0`, because each digit may appear only once per row. Deviations contribute to the loss `L_row`; in a solved Sudoku, `L_row → 0`.

## The Column Loss
The same process applies to columns:

<img width="30%" alt="image" src="https://github.com/user-attachments/assets/751ac262-ef72-4971-9617-8417843ae01a" />

```python
col_sum = P.sum(dim=0)          # (4,4) over rows -> cols x digits
L_col = ((col_sum - 1.0) ** 2).sum()
```
`col_sum` shows, for each column, the sum of probabilities per digit. For the first column (c = 0), the table looks like this:

| Digit | P[0,0,d] | P[1,0,d] | P[2,0,d] | P[3,0,d] | col_sum[0, d] = sum over rows |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.70 | 0.10 | 0.05 | 0.15 | **1.00** |
| 1 | 0.10 | 0.20 | 0.30 | 0.25 | **0.85** |
| 2 | 0.10 | 0.60 | 0.55 | 0.10 | **1.35** |
| 3 | 0.10 | 0.10 | 0.10 | 0.50 | **0.80** |

`col_sum[0] = [1.00, 0.85, 1.35, 0.80]`. The ideal value is `1.0` per digit because each digit may appear only once per column. Deviations feed into `L_col`; in a solved Sudoku, `L_col → 0`.

## The Block Loss
Each 2×2 block must also contain each digit exactly once. For a 4×4 Sudoku, `P` is reshaped to `2×2×2×2×4` and then summed over the inner block axes:

```python
blocks = P.view(2, 2, 2, 2, digits).sum(dim=(1, 3))  # (block_row, block_col, digit)
L_block = ((blocks - 1.0) ** 2).sum()
```

Consider the top-left block (block index `[0,0]`), which includes cells `(0,0), (0,1), (1,0), (1,1)`. Each of these four cells provides a probability distribution over the digits. `blocks[0,0]` adds the four cells per digit:

| Digit | P[0,0,d] | P[0,1,d] | P[1,0,d] | P[1,1,d] | blocks[0,0, d] = sum over 2×2 cells |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.70 | 0.05 | 0.10 | 0.15 | **1.00** |
| 1 | 0.10 | 0.05 | 0.20 | 0.25 | **0.60** |
| 2 | 0.10 | 0.80 | 0.60 | 0.10 | **1.60** |
| 3 | 0.10 | 0.10 | 0.10 | 0.50 | **0.80** |

`blocks[0,0] = [1.00, 0.60, 1.60, 0.80]`. Ideally `1.0` per digit ensures each block contains all digits once. Deviations determine the block loss `L_block`; in a solved Sudoku this value tends toward `0`.

# Step-by-Step Tutorial: Tensors and Operations
The following steps show how the tensors and operations described above can be implemented in PyTorch. All examples are chosen so they work immediately in a fresh Python session.

## 1) Create basic tensors
```python
import torch
import torch.nn.functional as F

# Fixed Sudoku clues (1-based input -> 0-based indices)
puzzle = torch.tensor([
    [0, 0, 0, 4],
    [0, 2, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 3, 0],
])

# Boolean mask indicating which cells are givens
givens_mask = puzzle > 0
```

## 2) Trainable logit tensor `Z`
```python
digits = 4
Z = torch.zeros(4, 4, digits, requires_grad=True)  # (row, col, digit)
```

## 3) Softmax with temperature for probabilities `P`
```python
temperature = 1.0
P = F.softmax(Z / temperature, dim=2)  # each cell sums to 1 over the digits
```

## 4) Loss functions for Sudoku rules
```python
# Enforce givens
one_hot_givens = F.one_hot((puzzle - 1).clamp(min=0), num_classes=digits)
L_given = ((P[givens_mask] - one_hot_givens[givens_mask]) ** 2).sum()

# Row and column rules
row_sum = P.sum(dim=1)               # (4, 4)
col_sum = P.sum(dim=0)               # (4, 4)
L_row = ((row_sum - 1.0) ** 2).sum()
L_col = ((col_sum - 1.0) ** 2).sum()

# Block rule for 4x4 (2x2 blocks)
blocks = P.view(2, 2, 2, 2, digits).sum(dim=(1, 3))  # (2, 2, 4)
L_block = ((blocks - 1.0) ** 2).sum()

# Total cost
loss = L_given + L_row + L_col + L_block
```

## 5) Optimization with Adam
```python
opt = torch.optim.Adam([Z], lr=0.05)
for step in range(500):
    opt.zero_grad()
    P = F.softmax(Z / temperature, dim=2)

    row_sum = P.sum(dim=1)
    col_sum = P.sum(dim=0)
    blocks = P.view(2, 2, 2, 2, digits).sum(dim=(1, 3))
    L_row = ((row_sum - 1.0) ** 2).sum()
    L_col = ((col_sum - 1.0) ** 2).sum()
    L_block = ((blocks - 1.0) ** 2).sum()

    L_given = ((P[givens_mask] - one_hot_givens[givens_mask]) ** 2).sum()
    loss = L_given + L_row + L_col + L_block
    loss.backward()
    opt.step()

solution = P.argmax(dim=2) + 1  # convert back to 1..4
print(solution)
```

## 6) Interpret the result
- `solution` contains the Sudoku numbers between 1 and 4.
- The sum of the losses `loss` should approach 0 when all rules are satisfied.
- Adjust `temperature` to control the “sharpness” of the softmax and stabilize optimization.
