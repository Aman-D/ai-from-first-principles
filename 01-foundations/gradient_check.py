
def f(w):
    return w ** 2

w = 3.0
epsilon = 0.0001

numerical_gradient = (
    f(w + epsilon) - f(w - epsilon)
) / (2 * epsilon)

analytical_gradient = 2 * w

print("numerical gradient:", numerical_gradient)
print("analytical gradient:", analytical_gradient)



def loss_fn(w):
    x = 2.0
    target = 10.0

    prediction = x * w
    return (prediction - target) ** 2


w = 3.0
x = 2.0
target = 10.0
epsilon = 0.0001

# Numerical gradient
numerical_gradient = (
    loss_fn(w + epsilon) - loss_fn(w - epsilon)
) / (2 * epsilon)

# Analytical gradient
prediction = x * w
analytical_gradient = 2 * (prediction - target) * x

print("numerical:", numerical_gradient)
print("analytical:", analytical_gradient)



def f(w, b):
    return w**2 + b**2

w = 3.0
b = 4.0

dw = 2 * w
db = 2 * b

print("df/dw:", dw)
print("df/db:", db)

# This is exactly what happens in a neural network with many weights and biases: every trainable parameter gets its own partial derivative.