## 线性神经网络

### 线性回归的从零开始实现

- 生成数据集
- 读取数据集
- 初始化模型参数
- 定义模型
- 定义损失函数loss function
- 定义优化算法（通常是sgd）
- 训练

#### 生成数据集

```python
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
```

### basic information

#### 1.TensorDataset

```python
dataset = data.TensorDataset(*data_arrays)
```



`TensorDataset` 是 PyTorch 中提供的一个数据集类，用于将多个张量（tensors）组合成一个数据集。每个张量都代表了数据集中的一个特征或标签。

在这个代码中，`data_arrays` 是一个包含多个张量的列表或元组。每个张量对应数据集中的一个特征或标签。

通过使用 `*` 操作符，将 `data_arrays` 列表或元组中的每个元素解包为单独的参数，传递给 `TensorDataset` 类的构造函数。这样就创建了一个 `TensorDataset` 对象，其中包含了所有特征和标签的数据集。

创建 `TensorDataset` 对象后，可以将其用作数据加载器（data loader）的参数，从而实现对数据的批量加载和处理。数据加载器是 PyTorch 提供的一个工具，用于方便地加载数据集，并支持批量处理、数据增强等功能。

总而言之，这行代码的作用是将多个特征和标签张量组合成一个 `TensorDataset` 对象，以便后续使用数据加载器进行数据的批量处理和训练

### softmax 回归框架

#### 库依赖

```python
import torch
from torch import nn
from d2l import torch as d2l
```

#### 数据集读取，批量大小设置

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

#### 初始化模型参数

```
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

#### loss function

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

#### 优化算法

```python
trainer = torch.optim.SGD(net.parameters(), lr=-0.1)
```

#### 训练

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

