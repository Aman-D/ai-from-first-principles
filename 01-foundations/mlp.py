#Multi-Layer Perceptron.

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3), #[input_feature, neurons]
    nn.ReLU(),
    nn.Linear(3, 1),
)

print(model)


x = torch.tensor([[2.0, 3.0]])

prediction = model(x)

print("input shape:", x.shape)
print("prediction:", prediction)
print("prediction shape:", prediction.shape)



X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [2.0],
])



print("X shape:", X.shape)
print("y shape:", y.shape)


print("predictions before training:")
print(model(X))


loss_fn = nn.MSELoss() # Mean Squared Error

predictions = model(X)
loss = loss_fn(predictions, y)

print("initial loss:", loss.item())


#Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(200):
    predictions = model(X)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print("step:", step, "loss:", loss.item())

print("predictions after training:")
print(model(X))