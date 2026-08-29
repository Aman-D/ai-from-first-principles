import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
target = torch.tensor(10.0)

learning_rate = 0.6

for step in range(20):
    prediction = x * w
    loss = (prediction - target) ** 2

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

    print(
        step,
        "w =", round(w.item(), 4),
        "prediction =", round(prediction.item(), 4),
        "loss =", round(loss.item(), 4),
    )