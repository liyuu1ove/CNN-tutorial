import torch
print("1")
# 测试CUDA
print("Support CUDA?:", torch.cuda.is_available())
x = torch.tensor([10.0])
x = x.cuda()
print(x)

y = torch.randn(2, 3)
y = y.cuda()
print(y)

z = x + y
print(z)

# 测试CUDNN
from torch.backends import cudnn
print("Support cudnn?:", cudnn.is_available())

print(torch.cuda.is_available())
print(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.version.cuda)
