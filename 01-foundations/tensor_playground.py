import torch

scalar = torch.tensor(7)

vector = torch.tensor([1, 2, 3])

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("Scalar:", scalar)
print("Shape:", scalar.shape)

print()

print("Vector:", vector)
print("Shape:", vector.shape)

print()

print("Matrix:")
print(matrix)
print("Shape:", matrix.shape)




print("\n--- Indexing ---")

print(matrix[0])
print(matrix[1])

print(matrix[0, 0])
print(matrix[0, 2])
print(matrix[1, 1])


print("\n--- Reshape ---")

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("Original:")
print(x)
print("Shape:", x.shape)

reshaped = x.reshape(3, 2)

print("\nReshaped:")
print(reshaped)
print("Shape:", reshaped.shape)


x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

weights = torch.tensor([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

output = x @ weights

print(output)
print(output.shape)



print("\n--- Broadcasting ---")
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

bias = torch.tensor([10.0, 20.0, 30.0])

print(x + bias)
print((x + bias).shape)