## 环境配置
下载和安装 [anaconda](https://anaconda.org/anaconda/conda) （网卡可选 [清华镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D) ）

安装好anaconda后启动Anaconda prompt并输入指令
```shell
(base) conda create -n pytorch_cpu python=3.9
(base) conda activate pytorch_cpu
(pytorch_cpu) pip3 install torch torchvision torchaudio
```
指令输入完成即可将conda加入python的运行环境，运行`python 3.9('pytorch_cpu')`即可运行


使用 [pytorch](https://pytorch.org/) python API构建MNIST-recognize CNN。本地CPU即可运行，无需担心配置问题

[google colab](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/0e6615c5a7bc71e01ff3c51217ea00da/tensorqs_tutorial.ipynb#scrollTo=Pzb1CuJSbIVT)提供在线运行功能

## 前置知识

### 张量
张量是机器学习中非常重要且基本的信息存储方式。简单的理解，张量是高维数组的昵称。例如，我们在C语言中声明了一个2d数组：
```C
int arr[m][n];
```
这是一个二维张量。比如说在CNN中，这个张量可以用作存储二维的黑白图片或者一个RGB图片的通道

我们也可以声明一个一维数组代表一个向量：
```C
int arr[m];
```

用pytorch语法声明一个随机的张量（具体用法可参考`tensor.py`）：
```python
shape = (2,3)
tensor=torch.rand(shape)
``` 
下面函数可用来查看张量属性：
```python
tensor.shape
tensor.dtype
tensor.device
```

这里的`device`方法是指张量在什么设备上计算

（后续可以使用`tensor.to(device)`方法选择在指定的设备上运行比如说特定的CPU或GPU）

可以用add_方法对张量进行修改:
```python
tensor.add_()
```
带_的方法表示它们将结果存储在**原始内存**中，而非新建，这有时是危险的。


### 线性代数基础
#### 矩阵乘法
![matmul](asset\Neural_net_layers_as_matrices_and_vectors.png)*from https://khalidsaifullaah.github.io/neural-networks-from-linear-algebraic-perspective*
```python
torch.matmul(tensor_a,tensor_b)
```
#### 转置
```python
tensor_a.T
```

更多方法在会在后续提到