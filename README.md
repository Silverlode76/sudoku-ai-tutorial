<img width="50%"  alt="image" align="right" src="https://github.com/user-attachments/assets/97b00694-5810-409a-8309-181d6328f322"/>



# sudoku-ai-tutorial
An educational project designed to teach beginners the fundamentals of tensors, logits, normalization, loss functions, and optimizers through a hands-on neural Sudoku example. Ideal for learners entering machine learning, PyTorch, and gradient-based reasoning

# Introduction 
This course aims to explain how a Sudoku riddle can also be solved with an optimizer approach. We will use therefore Tensor operations to define the needed loss functions for the optimizer.

## The Sudoku problem 
A Sudoku Riddle is solved if we have in every row numbers 1,2,3,4 and if we have in every column numbers 1,2,3,4. Additionally the 4 blocks need to contain also numbers 1,2,3,4. 
Mathematically this is expressed as follows:

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/2ce3e79d-d94a-4239-b4df-99845c8727d7" />

## The Sudoku Puzzle
Let’s assume we have a Sudoku puzzle where only 4 numbers are given. We can put all givens with its values in a matrix as shown below

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/a6b2e530-1f56-4b17-a2b6-0bebd7972f64" />

## The Sudoku Puzzle as Tensor 
Since we have created a matrix which represents our puzzle with the given number we need to define the matrix and the gins I tensors due to pytorch manipulation is done with tensors.
Due to tensor indices start with 0 we need to subtract 1 because our number from the riddle are between 1..4 which represents tensor indices 0..3
Table below shos the Sudoku Riddle, the Sudoku Tensor and the logical Givens

| Step | Description | Image |
|------|-------------|--------|
| 1 | Input Sudoku | <img width="208" height="148" alt="image" src="https://github.com/user-attachments/assets/4006291d-f159-4fc7-ba38-6f30fbd381c5" />|
| 2 | Input Sudoku Tensor | <img width="200" height="130" alt="image" src="https://github.com/user-attachments/assets/634c4231-ad17-4bac-8806-b02410a738a5" />|
| 3 | Input Givens | <img width="218" height="130" alt="image" src="https://github.com/user-attachments/assets/04be267b-fa61-46b4-9479-b6e53dc30574" />|

# Preparing the Z Tensor (The Puzzle Tensor)
Since we have the Sudoku Tensor and the Givens we can create now our Z Tensor which contains the digits for our Loss Functions.
Now let’s add the dimension for the digits that we have a tensor which represents beside row and column also the value of the number here the digits from 0..3
In our Z cube would look like follows (red and green are givens from our riddle). Each digit we already have as givens is marked with 5.
| Tensor Z | Description |
|------|-------------|
|<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/c4441557-83a9-4209-b140-8f05d1fa4501" />|<img width="1186" height="666" alt="image" src="https://github.com/user-attachments/assets/2f62fa3a-2a6e-431b-bd5e-b36487ba09bd" />
|














