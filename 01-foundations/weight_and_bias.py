import torch

x = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

target = torch.tensor(7.0)

learning_rate = 0.1

for step in range(10):
    prediction = x * w + b
    loss = (prediction - target) ** 2

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(
        step,
        "w =", round(w.item(), 4),
        "b =", round(b.item(), 4),
        "prediction =", round(prediction.item(), 4),
        "loss =", round(loss.item(), 4),
    )