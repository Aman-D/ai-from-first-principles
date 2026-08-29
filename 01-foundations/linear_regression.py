# 01-foundations/linear_regression.py

# x = 2.0
# target = 10.0

# w = 3.0
# learning_rate = 0.01

# prediction = x * w
# loss = (prediction - target) ** 2

# print("prediction:", prediction)
# print("loss:", loss)


# gradient = 2 * (prediction - target) * x
# print("gradient:", gradient)


# w = 3.16

# prediction = x * w
# loss = (prediction - target) ** 2


# print("new prediction:", prediction)
# print("new loss:", loss)


x = 2.0
target = 10.0

w = 3.0
learning_rate = 0.1

for step in range(20):
    prediction = x * w
    loss = (prediction - target) ** 2

    gradient = 2 * (prediction - target) * x

    w = w - learning_rate * gradient

    print(
        step,
        "w =", round(w, 4),
        "prediction =", round(prediction, 4),
        "loss =", round(loss, 4),
    )