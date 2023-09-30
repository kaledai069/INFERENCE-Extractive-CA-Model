import torch

# Your input tensor with token IDs
input_tensor = torch.tensor([
    [101, 6976, 13109, 3771, 1006, 2773, 13068, 1007, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Add more rows as needed
])

# Create a tensor of the same shape with boolean values
boolean_tensor = (input_tensor != 0)

# Print the resulting boolean tensor
print(boolean_tensor)