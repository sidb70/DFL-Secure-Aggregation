import torch

# Create a sample 2D tensor
tensor = torch.randn(5, 3)
print("Original tensor:")
print(tensor)

# Sort each column
sorted_tensor, indices = torch.sort(tensor, dim=0)

print("\nSorted tensor (by columns):")
print(sorted_tensor)

print("\nIndices:")
print(indices)