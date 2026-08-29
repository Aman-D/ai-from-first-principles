import torch

x = torch.tensor([2.0, 3.0])

# w = torch.tensor([0.5, 1.0])
w = torch.tensor([-0.5, -1.0])

b = torch.tensor(1.0)

weighted_sum = torch.dot(x, w) + b

print("weighted sum:", weighted_sum.item())


relu_output = torch.relu(weighted_sum)
tanh_output = torch.tanh(weighted_sum)

print("ReLU:", relu_output.item())
print("tanh:", tanh_output.item())






# z = torch.tensor(-3.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

# ReLU gradient
relu_output = torch.relu(z)
relu_output.backward()

print("ReLU output:", relu_output.item())
print("ReLU gradient:", z.grad.item())

# Reset gradient
z.grad.zero_()

# tanh gradient
tanh_output = torch.tanh(z)
tanh_output.backward()

print("tanh output:", tanh_output.item())
print("tanh gradient:", z.grad.item())