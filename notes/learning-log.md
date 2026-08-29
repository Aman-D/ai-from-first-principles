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
- Numerical derivatives using a small epsilon
- Analytical vs numerical gradient checking
- Chain rule as multiplication of local effects
- Partial derivatives for multiple parameters
- Single neuron as weighted sum + bias + activation
- ReLU behavior for positive vs negative inputs
- tanh output range `[-1, 1]` and saturation
- ReLU vs tanh gradient behavior

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
- Epsilon = a tiny step used to estimate a derivative numerically
- Partial derivative = change one parameter while treating the others as fixed
- ReLU: negative input -> output 0, gradient 0; positive input -> output x, gradient 1
- tanh: preserves sign and squashes outputs to `[-1, 1]`; near ±1 it saturates and gradients become very small

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
- Built `01-foundations/gradient_check.py`
- Verified `f(w)=w^2` numerical gradient ≈ analytical gradient at `w=3`
- Verified the Day 3 linear-regression gradient numerically (`≈ -16`) against the analytical gradient (`-16`)
- Practiced partial derivatives with `f(w,b)=w^2+b^2`
- Built `01-foundations/neuron.py`
- Compared ReLU and tanh outputs at positive and negative neuron outputs
- Compared ReLU and tanh gradients at `z=-3` and `z=1`
- Observed tanh saturation near `-1` and ReLU's zero gradient for negative inputs

### Current position

- Day 1 complete
- Day 2 complete
- Day 3 complete: linear regression from scratch without autograd
- Day 4 complete: derivative intuition, chain rule, partial derivatives, and numerical gradient checking
- Day 5 complete: single neuron, ReLU/tanh outputs, gradients, and saturation intuition
- Day 6 is next: build and train a small MLP

### Next

Day 6 — build and train a small multilayer perceptron.
