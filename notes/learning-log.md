## 2026-08-23

### Learned

- Tensor shapes and dimensions
- Indexing and reshape
- Matrix multiplication shape rule
- Basic neuron/layer intuition
- Broadcasting

### Key mental models

- `[A, B] @ [B, C] -> [A, C]`
- Linear layer: `X @ W + bias`
- `W.shape = [input_features, number_of_neurons]`
- Reshape must preserve total element count

### Still unclear

- None / add anything you want to revisit

### Next

Linear algebra intuition for neural networks

---

## 2026-08-29

### Learned

- Vectors and dot products
- Matrix multiplication as many neuron dot products
- Layers, neurons, weights, and bias
- ReLU and why activations add non-linearity
- Forward pass through a tiny 2-layer network
- Loss as a measure of prediction error
- Gradient intuition and gradient descent
- Learning-rate behavior: slow, convergent, and divergent cases
- PyTorch autograd: `requires_grad`, `loss.backward()`, `.grad`
- Why `torch.no_grad()` is used during parameter updates
- Why gradients must be cleared with `zero_()`
- Trainable parameters vs activations vs gradients
- Backpropagation intuition through multiple parameters

### Key mental models

- One neuron = one dot product + bias (+ activation)
- One layer = many neurons evaluated together
- `[batch, input_features] @ [input_features, neurons] -> [batch, neurons]`
- Parameters are learned; activations are computed; gradients tell parameters how to change
- Forward pass: input -> prediction -> loss
- Backward pass: loss -> gradients for parameters
- Gradient = sensitivity of loss to a parameter
- Gradient descent update: `new_weight = old_weight - learning_rate * gradient`
- Learning rate = step size; too large can overshoot and diverge
- Backpropagation = chain rule applied backward through the computation graph

### Experiments completed

- Trained a one-parameter model toward `w = 5`
- Compared learning rates `0.01`, `0.1`, and `0.6`
- Observed stable convergence vs divergence
- Trained weight and bias together for `prediction = x*w + b`
- Inspected separate gradients for multiple trainable parameters
- Built `01-foundations/linear_algebra.py` with dot-product, weighted-sum, and multi-neuron matrix examples
- Built `01-foundations/linear_regression.py` without autograd
- Implemented prediction, squared-error loss, manual gradient calculation, and parameter updates
- Compared learning rates `0.001`, `0.01`, and `0.1`
- Observed `0.001` learning slowly, `0.01` converging steadily, and `0.1` reaching the optimum quickly for the toy problem

### Current position

- Day 1 complete
- Day 2 complete
- Day 3 complete: linear regression from scratch without autograd
- Some Day 4/5/6 concepts have been previewed ahead of schedule, but those issues remain open until their build artifacts and definitions of done are satisfied

### Next

Day 4 — derivatives, chain rule, and gradient checking.
