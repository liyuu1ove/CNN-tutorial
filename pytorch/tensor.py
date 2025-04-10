import torch

dtype = torch.float
device = torch.device("cpu")

# different shape
shape = (2,3)
#shape = (2,3,4) 
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#Attributes of a Tensor 
rand_tensor = torch.rand(2,3)
print(f"Shape of tensor: {rand_tensor.shape}")
print(f"Datatype of tensor: {rand_tensor.dtype}")
print(f"Device tensor is stored on: {rand_tensor.device}")

print(f"Before: \n {rand_tensor} \n")
rand_tensor.add_(1)
print(f"Random Tensor: \n {rand_tensor} \n")