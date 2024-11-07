### tensor.numel()

- 返回一个张量（`Tensor`）中元素的总数。
- 与张量形状无关

### torch.eye()

- 创建一个二维的单位矩阵（identity matrix），即对角线上的元素为 `1`，其余元素为 `0`。这是一个常用的线性代数操作。

- ```python
  torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  ```

  - `n`：单位矩阵的行数。
  - `m`（可选）：单位矩阵的列数。如果不指定，则默认为与 `n` 相同，即创建一个 `n x n` 的方阵。
  - `out`（可选）：可以指定输出张量。
  - `dtype`（可选）：指定数据类型（如 `torch.float32`）。
  - `layout`（可选）：指定布局（默认是 `torch.strided`）。
  - `device`（可选）：指定设备（如 `torch.device('cuda')`）。
  - `requires_grad`（可选）：是否记录梯度（默认为 `False`）。

### torch.rand(4, 5)

- 生成一个形状为 `(4, 5)`的张量，其中包含从均匀分布 `[0,1)`中随机采样的浮点数。生成的张量的每个元素都是随机的，并且位于 `0`（包含）到 `1`（不包含）之间

- ```python
  torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  ```

  `*size`：张量的形状。在这个例子中，`4, 5` 表示生成一个 4 行 5 列的矩阵。

  `out`（可选）：用于存储输出的张量。

  `dtype`（可选）：指定数据类型（如 `torch.float32`）。

  `layout`（可选）：指定张量布局（默认是 `torch.strided`）。

  `device`（可选）：指定生成张量的设备（如 `'cpu'` 或 `'cuda'`）。

  `requires_grad`（可选）：是否需要梯度（默认为 `False`）。

### Pytorch commonly used datatypes

- `torch.float32`：标准浮点类型，存储可以学习的参数，网络的激活函数。激活所有的算数表达式都用的float32
- `torch.int64`：通常用来存储indices
- `torch.bool`：存储布尔值，0表示false，1表示true
- `torch.float16`：用来做mixed-precision arithmetic



