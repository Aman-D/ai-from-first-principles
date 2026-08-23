# First Week — Execution Plan

Detailed tasks should become GitHub Issues and stay adaptive.

## Day 1 — Environment & Tensor Fundamentals
- Set up Python environment
- Install PyTorch
- Tensor creation, shape, dtype, indexing, reshape
- Matrix multiplication and broadcasting
- Deliverable: `01-foundations/tensor_playground.py`
- Experiment: identify and fix 3 shape mismatch examples
- Commit: `feat: add pytorch tensor playground`

## Day 2 — Linear Algebra Intuition
- vectors, matrices, dot products
- implement small matrix operations
- connect matrix multiplication to a neuron/layer
- Deliverable: `01-foundations/linear_algebra.py`
- Commit: `feat: add linear algebra exercises for neural networks`

## Day 3 — Linear Regression From Scratch
- prediction, MSE, parameter update
- no PyTorch autograd
- Deliverable: working regression training loop
- Experiment: compare 3 learning rates
- Commit: `feat: train linear regression without autograd`

## Day 4 — Derivatives & Gradients
- derivative intuition, partial derivatives, chain rule
- numerical gradient checking
- Deliverable: gradient-check script + notes
- Commit: `experiment: verify analytical gradients numerically`

## Day 5 — Build a Neuron
- weighted sum, bias, activation
- Deliverable: single neuron implementation
- Experiment: ReLU vs tanh
- Commit: `feat: implement neuron from first principles`

## Day 6 — Build an MLP
- layers, forward pass, loss
- use PyTorch autograd initially
- Deliverable: small classifier
- Commit: `feat: train first multilayer perceptron`

## Day 7 — Review & Retrospective
- explain forward → loss → backward → update
- clean repository
- write weekly summary
- list confusions for Week 2
- Commit: `docs: add week one neural network retrospective`
