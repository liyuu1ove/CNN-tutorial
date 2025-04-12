import torch 
import torch.nn as nn 

# Define the input tensor 
input_tensor = torch.tensor( 
	[ 
		[1, 1, 2, 4], 
		[5, 6, 7, 8], 
		[3, 2, 1, 0], 
		[1, 2, 3, 4] 
	], dtype = torch.float32) 

# Reshape the input_tensor 
input_tensor = input_tensor.reshape(1, 1, 4, 4) 

# Initialize the Max-pooling layer with kernel 2X2 and stride 2 
pool = nn.MaxPool2d(kernel_size=2, stride=2) 

# Apply the Max-pooling layer to the input tensor 
output = pool(input_tensor) 

# Print the output tensor 
print(f'Output Shape :',output.shape) 
print(f'maxpooling\n',output)