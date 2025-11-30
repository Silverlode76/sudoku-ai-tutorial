<img width="50%"  alt="image" align="right" src="https://github.com/user-attachments/assets/97b00694-5810-409a-8309-181d6328f322"/>



# sudoku-ai-tutorial
An educational project designed to teach beginners the fundamentals of tensors, logits, normalization, loss functions, and optimizers through a hands-on neural Sudoku example. Ideal for learners entering machine learning, PyTorch, and gradient-based reasoning

# Introduction 
This course aims to explain how a Sudoku puzzle can also be solved with an optimizer approach. We will use therefore Tensor operations to define the needed loss functions for the optimizer.

## The 4x4 Sudoku problem 
A Sudoku puzzle is solved if we have in every row numbers 1,2,3,4 and if we have in every column numbers 1,2,3,4. Additionally the 4 blocks need to contain also numbers 1,2,3,4. 
Mathematically this is expressed as follows:

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/2ce3e79d-d94a-4239-b4df-99845c8727d7" />

## The 4x4 Sudoku Puzzle
Let’s assume we have a Sudoku puzzle where only 4 numbers are given. We can put all givens with its values in a matrix as shown below

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/a6b2e530-1f56-4b17-a2b6-0bebd7972f64" />

## The 4x4 Sudoku Tensors
Sudoku Puzzle as Tensor 
Since we have created a matrix which represents our puzzle with the given number we need to define the matrix and the givens in tensors due to pytorch manipulation is done with tensors.
Due to tensor indices start with 0 we need to subtract 1 because our number from the puzzle are between 1..4 which represents tensor indices 0..3
Table below shows the Sudoku puzzle, the Sudoku Tensor and the logical Givens

| Step | Description | Image |
|------|-------------|--------|
| 1 | Input Sudoku | <img width="208" height="148" alt="image" src="https://github.com/user-attachments/assets/4006291d-f159-4fc7-ba38-6f30fbd381c5" />|
| 2 | Input Sudoku Tensor | <img width="200" height="130" alt="image" src="https://github.com/user-attachments/assets/634c4231-ad17-4bac-8806-b02410a738a5" />|
| 3 | Input Givens | <img width="218" height="130" alt="image" src="https://github.com/user-attachments/assets/04be267b-fa61-46b4-9479-b6e53dc30574" />|

# Preparing the Z Tensor (The Puzzle Tensor)
Since we have the Sudoku Tensor and the Givens we can create now our Z Tensor which contains the digits for our Loss Functions.
Now let’s add the dimension for the digits that we have a tensor which represents beside row and column also the value of the number here the digits from 0..3
In our Z cube would look like follows (red and green are givens from our puzzle). Each digit we already have as givens is marked with 5.
| Tensor Z | Description |
|------|-------------|
|<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/c4441557-83a9-4209-b140-8f05d1fa4501" />|<img width="1186" height="666" alt="image" src="https://github.com/user-attachments/assets/2f62fa3a-2a6e-431b-bd5e-b36487ba09bd" />

# Preparing the Optimizer 
As Optimizer we use Adam Optimizer. The Input is our Tensor Z  
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

Adam is the ideal optimizer here because:
- it adapts the learning rate per tensor element
- it combines Momentum and RMSProp, providing stability
- it handles flat regions and steep regions effectively
- it converges much faster than SGD
- it works very well with softmax and temperature annealing

## Temperature Annealing / Sofmax

The P is the probalility denysity of Tensor Z.
Example of Softamx():

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
The Tensors Z and P has the dimensions row (dim=0), col (dim=1) and digit (dim =2). In the function above we do Softmax "over thh digits" therefore we have to put dim=2.
This means that if we sum up for every element as P(0,0, :) we get sum = 1 which is shown below

| Tensor Z | Tensor P |
|------|-------------|
| <img width="80%" alt="image" src="https://github.com/user-attachments/assets/18fbd8c2-362c-4191-b40a-44600da8ff15" /> | <img width="80%" alt="image" src="https://github.com/user-attachments/assets/88e32fa6-169f-4b44-ae4a-f6e63770ad17" />|

Let's consider slice = 0 (or row = 0). The Sum over the digits is always 1.0. 

| Z-Vector (Logits)                            | P-Vector (Softmax)                 | Sum(P) |
| -------------------------------------------- | ---------------------------------- | ------ |
| `[5.0, -5.0, -5.0, -5.0]`                    | `[0.9424, 0.0202, 0.0202, 0.0172]` | `1.0`  |
| `[-0.0059, -0.0025, 0.0057, -0.0039]`        | `[0.2296, 0.2698, 0.2707, 0.2298]` | `1.0`  |
| `[4.7337e-04, -2.0038e-03, 7.8445e-03, ...]` | `[0.2298, 0.2694, 0.2705, 0.2303]` | `1.0`  |
| `[-5.0, -5.0, -5.0, 5.0]`                    | `[0.0172, 0.0202, 0.0202, 0.9424]` | `1.0`  |

# The Lossfunction 

## The Row Loss
For each row we have now to determine the loss which we do with:

<img width="30%" alt="image" src="https://github.com/user-attachments/assets/76bff619-6583-46b4-bdea-beb337c8de4a" />

```python
row_sum = P.sum(dim=1)          # (4,4) over columns -> rows x digits
L_row = ((row_sum - 1.0) ** 2).sum()
```
The Tensor row_sum has the dimensions (row, digits) after the operation from above.
Let's take for demonstration the first slice = 0 and let's calculate the sum over the coliums

| P-Tensor (0,c, digits)                       | row_sum                |
| -------------------------------------------- | ---------------------- |
|[0.9424, 0.0202, 0.0202, 0.0172] | 0.9424 + 0.2304 + 0.2304 + 0.0172 = 1.4204 |
|[0.2304, 0.2706, 0.2695, 0.2295] | 0.0202 + 0.2706 + 0.2710 + 0.0202 = 0.5819 |
|[0.2304, 0.2710, 0.2700, 0.2285] | 0.0202 + 0.2695 + 0.2700 + 0.0202 = 0.5799 | 
|[0.0172, 0.0202, 0.0202, 0.9424] | 0.0172 + 0.2295 + 0.2285 + 0.9424 = 1.4176 |

row_sum[0] = [1.4204, 0.5819, 0.5799, 1.4176]
So if we apply here the Soduko Rule that the sum for each row must be 1 then we see here that for digit 0 & 3 we have 1.42 and 1.41 which is too high and for the other digits it's too low.
In the ideal case and when the riddle is solved we would get with L_row = ((row_sum - 1.0) ** 2).sum() = 0

## The Colum Loss
What we did for the rows had now also be applied to the columns with:

<img width="30%" alt="image" src="https://github.com/user-attachments/assets/751ac262-ef72-4971-9617-8417843ae01a" />

```python
col_sum = P.sum(dim=0)          # (4,4) over rows -> cols x digits
L_col = ((col_sum - 1.0) ** 2).sum()
```






















