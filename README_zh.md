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

## 卷积神经网络CNN
卷积神经网络（CNN）是人工神经网络的高级版本，主要用于从网格状矩阵数据集中提取特征。这对于图像或视频等可视化数据集特别有用，其中数据模式起着至关重要的作用。
Convolutional Neural Network (CNN) is an advanced version of artificial neural networks,primarily designed to extract features from grid-like matrix datasets. This is particularly useful for visual datasets such as images or videos, where data patterns play a crucial role.
### CNN 结构

![structure](asset\structure.jpeg)
*from https://www.geeksforgeeks.org/apply-a-2d-max-pooling-in-pytorch/*

### 1. 输入层（Input Layer）
作用：接收原始数据（如RGB图像的三通道矩阵 [height, width, channels]）。

### 2. 卷积层（Convolutional Layer）
* 核心思想是通过局部感知和参数共享高效提取输入数据的空间特征

#### 卷积核（Filter/Kernel）
>形状：通常为小尺寸正方形（如3×3、5×5），深度与输入通道数相同。
>
>示例：对RGB图像（3通道），一个3×3卷积核实际尺寸是 3×3×3。
>
>作用：每个卷积核检测一种特定局部特征（如边缘、纹理、颜色渐变）。

#### 局部感受野（Local Receptive Field）
>原理：卷积核每次只处理输入的一小块区域（如3×3窗口），而非全图。
>
>优势： 减少参数量（相比全连接层）。
>
>保留空间局部相关性（图像中相邻像素关系更紧密）。

#### 偏置（Bias）
>作用：
>
>1) 偏移特征图：Bias为卷积输出的每个特征图（Feature Map）的所有位置添加一个常数偏移量。
>
>2) 提升模型灵活性：允许卷积层在无输入（或输入全为零）时仍能产生非零输出（由Bias决定）。
>
>3) 与激活函数配合：Bias帮助调整神经元的激活阈值（尤其是ReLU等激活函数中“何时激活”）。

![convolution](asset\convolution.gif)
*from https://www.geeksforgeeks.org/apply-a-2d-max-pooling-in-pytorch/*

代码示例可参考 `convolution.py`

### 池化层（Pooling Layer）
>用于降低特征图的空间尺寸（降采样），同时保留重要信息。
>
>降维减参：减少计算量和内存消耗。
>
>平移不变性：使网络对输入的小幅位移、旋转更鲁棒。
>
>防止过拟合：通过压缩特征图，抑制噪声和冗余细节

![maxpooling](asset\maxpooling.png)
*from https://www.geeksforgeeks.org/apply-a-2d-max-pooling-in-pytorch/*

代码示例可参考 `maxpooling.py`
### 激活函数（Activation Function）
>作用：引入非线性，使网络能拟合复杂函数。
>
>常用函数：
>1) ReLU：f(x) = max(0, x)（缓解梯度消失，计算高效）。
>
>2) LeakyReLU/Swish：解决ReLU的“神经元死亡”问题。
### 损失函数（Loss Function）
>用于量化模型预测结果与真实值之间差异的函数。
>
>损失函数的核心作用
>1) 目标导向：为模型提供明确的优化目标（最小化损失）。
>
>2) 反馈信号：通过梯度指导参数更新（如反向传播）。
>
>3) 评估性能：衡量模型在训练/测试集上的表现（非唯一指标）。
>
>有多种函数需更具具体问题选择合适的函数
### 优化器（Optimizer）
>用于调整模型参数以最小化损失函数的算法，其核心目标是通过梯度下降的变体方法高效更新权重。
>
>优化器的核心作用
参数更新：根据损失函数的梯度调整模型参数（如权重W 和偏置b）。
>
>加速收敛：通过自适应学习率或动量等技术提高训练效率。
>
>避免局部最优：帮助跳出鞍点或平坦区域。