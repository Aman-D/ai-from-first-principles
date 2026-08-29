import torch

# 1. Vector / dot product
x = torch.tensor([2.0, 3.0])
w = torch.tensor([4.0, 5.0])

dot = torch.dot(x, w)
print("dot product:", dot.item())

# 2. One neuron = weighted sum
bias = torch.tensor(1.0)
output = dot + bias
print("one neuron output:", output.item())

# 3. Multiple neurons = matrix multiplication
X = torch.tensor([
    [2.0, 3.0],
    [1.0, 4.0],
])

W = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

b = torch.tensor([1.0, 1.0, 1.0])

layer_output = X @ W + b

print("X shape:", X.shape)
print("W shape:", W.shape)
print("layer output:")
print(layer_output)