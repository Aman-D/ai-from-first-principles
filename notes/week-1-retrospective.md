# Week 1 Retrospective

## What I can explain now

### Forward pass

A neural network takes an input, combines input values with trainable weights and biases, applies activation functions, and produces a prediction.

Mental model:

`input -> weighted sum + bias -> activation -> next layer -> prediction`

For a batch:

`[batch, input_features] @ [input_features, neurons] -> [batch, neurons]`

### Loss

Loss measures how wrong the prediction is compared with the target. For regression, squared error / MSE is a simple example.

### Gradients and backpropagation

A gradient tells how sensitive the loss is to a small change in a trainable parameter.

Backpropagation applies the chain rule backward through the computation graph to calculate gradients for the trainable weights and biases.

### Parameter update

Gradient descent uses the gradients to move parameters toward lower loss:

`new_parameter = old_parameter - learning_rate * gradient`

The learning rate controls the size of the update.

### Full training loop

`forward -> loss -> clear old gradients -> backward -> update parameters -> repeat`

## Week 1 checkpoint

Score: **16/17**

The core mental model is solid. Remaining gaps are mainly terminology and precision rather than missing concepts.

### Small corrections to keep reinforcing

- A neuron output before activation is the **sum** of weighted inputs plus bias, not the individual multiplications.
- Activation functions introduce **nonlinearity**; stacking linear layers without activations is equivalent to one larger linear transformation.
- A gradient is best described as the sensitivity/slope of loss with respect to a parameter, rather than simply how much that parameter "contributed" to the loss.
- `zero_grad()` should happen before the next backward pass so old gradients do not accumulate into the new step.

## Three strongest learnings

1. **Shapes, neurons, and layers** — I can reason about matrix multiplication, batch shapes, and how the number of neurons determines layer output size.
2. **The learning loop** — I can explain prediction -> loss -> gradients -> parameter update without depending on PyTorch jargon.
3. **Gradient intuition** — I understand gradient sign, learning rate, gradient descent, chain rule, and why backpropagation is needed.

## Three questions for Week 2

1. How does backpropagation work internally when a computation graph becomes larger and branches into many operations?
2. How are gradients accumulated and propagated through every node when building an autograd engine ourselves?
3. How do initialization, dataset splits, overfitting, and training dynamics affect whether a neural network actually generalizes?

## Experiments completed

- Tensor shape and broadcasting exercises
- Dot products and matrix multiplication
- Linear regression without autograd
- Learning-rate comparison
- Numerical vs analytical gradient checking
- Partial derivatives and chain rule exercises
- Single neuron with ReLU and tanh
- Activation saturation experiments
- First MLP trained with PyTorch autograd and SGD

## Week 1 conclusion

The important result is not memorizing APIs. I can now tell the story of how a basic neural network learns:

`input -> prediction -> loss -> gradients -> parameter updates -> better prediction`

Week 2 should deepen the mechanics behind that story, especially computation graphs, backpropagation, and autograd from first principles.
