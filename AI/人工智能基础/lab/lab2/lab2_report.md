# 人工智能基础 Lab 2 Report

>PB21111653
>
>李宇哲

##  2.1 传统机器学习

### 2.1.1 决策树

#### 原理

决策树算法的伪代码如下

![image-20240630165401438](./assets/image-20240630165401438.png)

#### 代码实现

##### 决策树结点

加入新的类 `TreeNode` 用于表示 决策树结点

```python
class TreeNode:
    def __init__(self, feature: int , threshold: Optional[float] = None, values: Optional[dict] = None, label: int = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.values = values if values else None
        self.label = label

    def is_leaf(self) -> bool:
        return self.label is not None
    
    @staticmethod
    def leaf(label:int) -> 'TreeNode':
        return TreeNode(feature=-1, label=label)
    
    def __repr__(self):
        if self.is_leaf():
            return f"Leaf: {self.label}"
        return f"Node: {self.feature}, {self.threshold}"
```

对于决策树，需要可以计算 信息熵 和 信息增益的接口函数，同时需要提供根据样本个数进行划分的接口

##### 信息熵和信息增益

信息熵定义为
$$
Ent(D) = - \sum_{k=1}^{\gamma} p_k \log_2 p_k
$$
信息增益定义为
$$
Gain(D,a) = Ent(D) - \sum_{v=1}^{V}\frac{|D_v|}{|D|}Ent(D^v)
$$

```python
def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

    def _information_gain(self, y, y_left, y_right):
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        return self._entropy(y) - (n_left/n * self._entropy(y_left) + n_right/n * self._entropy(y_right))
```

##### 最优划分

选择最优的结点，根据乜咯特征进行一遍搜索遍历，计算每个结点的信息增益，选择信息增益最大的那个特征

本质是一遍搜索

```python
def _best_split(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        # return: best_split_feature, best_split_value
        best_gain = -1
        best_split_feature, best_split_value = None, None

        for feature in X.columns:
            values = np.unique(X[feature])
            if X[feature].dtype.kind in 'bifc':
                values = (values[1:] + values[:-1]) / 2.0
            
            for value in values:
                y_left = y[X[feature] <= value]
                y_right = y[X[feature] > value]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split_feature = feature
                    best_split_value = value
        return best_split_feature, best_split_value
```

##### 连续值的处理

在最佳划分中，对连续值类型的变量进行二分处理

```python
if X[feature].dtype.kind in 'bifc':
                values = (values[1:] + values[:-1]) / 2.0
```

##### 生成决策树

决策树的算法流程

- 检查样本的标签是否相同，如果有相同表现的样本，创建一个叶节点，标签为样本的类别
- 检查是否有特征可以用于分裂
- 用`best_split`寻找最佳分裂特征和分裂值
- 进行分裂后，递归生成左右子树

```python
def Tree_Generate(self, X: pd.DataFrame, y: np.ndarray) -> TreeNode:
        # return: tree
        # tree = {'split_feature': split_feature, 'split_value': split_value, 'left': left_tree, 'right': right_tree}
        if len(np.unique(y)) == 1:
            return TreeNode.leaf(int(y[0]))

        if X.shape[1] == 0:
            return TreeNode.leaf(int(np.argmax(np.bincount(y))))
        
        split_feature, split_value = self._best_split(X, y)
        if split_feature is None:
            return TreeNode.leaf(int(np.argmax(np.bincount(y))))
        
        left_mask = X[split_feature] <= split_value
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        children = {
            'left': self.Tree_Generate(X_left, y_left),
            'right': self.Tree_Generate(X_right, y_right)
        }
        return TreeNode(split_feature, split_value, children, None)
```

##### 用决策树进行预测

```python
def tree_predict(self, X: pd.DataFrame, node: TreeNode):
        # X: [n_samples_test, n_features], node: TreeNode
        # return: y: [n_samples_test, ]
        if node.is_leaf():
            return np.array([node.label] * X.shape[0])
        left_mask = X[node.feature] <= node.threshold
        right_mask = ~left_mask
        X_left, X_right = X[left_mask], X[right_mask]
        y = np.zeros(X.shape[0])
        y[left_mask] = self.tree_predict(X_left, node.values['left'])
        y[right_mask] = self.tree_predict(X_right, node.values['right'])
        return y
```

#### 运行结果

在测试集上的预测结果为

![image-20240630174859855](./assets/image-20240630174859855.png)

### 2.1.2 PCA与K-means

#### 原理

##### PCA主成分分析

![image-20240630174957601](./assets/image-20240630174957601.png)

对所有样本中心化，然后计算协方差矩阵后，找到前K个特征值组成的特征向量充当后面K-means算法的主成分

##### K-means

伪代码如下

![image-20240630175109121](./assets/image-20240630175109121.png)

进行多轮迭代

- 计算机样本距离，确定每个样本的cluster
- 更新中心点

#### 代码实现

##### PCA部分

对PCA主成分分析，主要需要计算主成分的特征值和对应的特征向量，因此需要额外两个数据域来保存相关信息

```python
def __init__(self, n_components:int=2, kernel:str="rbf") -> None:
    # 主成分数量
    self.n_components = n_components
    # 核函数
    self.kernel_f = get_kernel_function(kernel)
    # kernel function type
    self.kernel = kernel
    # 尚未拟合的数据
    self.X_fit = None
    # 主成分的特征向量
    self.alpha = None
    # 主成分的特征值
    self.lambdas = None
    # ...
```

`fit`函数计算主成分的特征向量，更新两个数据域，以便在 transform中使用

```python
def fit(self, X:np.ndarray):
    # X: [n_samples, n_features]
    # TODO: implement PCA algorithm
    self.X_fit = X
    K = self.kernel_f(X, X)
    n = X.shape[0]

    # centering the kernel matrix
    one_n = np.ones((n, n)) / n
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(K)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    self.lambdas = eigvals[:self.n_components]
    self.alpha = eigvecs[:, :self.n_components]
```

计算流程遵循

- 计算核函数
- 中心化
- 求解特征向量

其中核函数定义如下

>核函数定义
>
>```python
>def get_kernel_function(kernel:str):
>    # TODO: implement different kernel functions 
>    if kernel == "linear":
>        return lambda x, y: np.dot(x, y.T)
>    elif kernel == "poly":
>        return lambda x, y, p = 3: (np.dot(x, y.T) + 1) ** p
>    elif kernel == "rbf":
>        return lambda x, y, sigma=5.0: np.exp(-np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) ** 2  / (2 * sigma ** 2))
>    else:
>        raise ValueError(f"Kernel {kernel} is not supported.")
>```

我们这里选择 rbf

>径向基函数核 RBF Kernel
>
>公式 $K(x,y) = \exp (- \frac{|| x-y||^2}{2 \sigma^2})$

transofrm部分

直接将数据用于低维度即可

```python
def transform(self, X:np.ndarray):
        K = self.kernel_f(X, self.X_fit)
        return K.dot(self.alpha) / np.sqrt(self.lambdas * X.shape[0])
```

##### K-means部分

K-means算法流程

- 初始化中心点

- 多轮迭代，每轮执行

  - 确定cluster个数

  - 根据上一轮的结果更新中心点

`fit`函数

```python
# k-means clustering
def fit(self, points):
    # points: (n_samples, n_dims,)
    # TODO: Implement k-means clustering
    self.initialize_centers(points)
    for _ in range(self.max_iter):
        old_centers = self.centers.copy()
        self.assign_points(points)
        self.update_centers(points)
        if np.all(old_centers == self.centers):
            break
```

`assign_points`

```python
# Assign each point to the closest center
    def assign_points(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        n_samples, n_dims = points.shape
        self.labels = np.zeros(n_samples)
        # TODO: Compute the distance between each point and each center
        # and Assign each point to the closest center
        for i in range(n_samples):
            distances = np.linalg.norm(points[i] - self.centers, axis=1)
            self.labels[i] = np.argmin(distances)
    
        return self.labels
```

`update_centers`

```python
def update_centers(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Update the centers based on the new assignment of points
        for k in range(self.n_clusters):
            cluster_points = points[self.labels == k]
            if len(cluster_points) > 0:
                self.centers[k] = cluster_points.mean(axis=0)
```

#### 运行结果

运行结果如下

![image-20240630205827591](./assets/image-20240630205827591.png)

## 2.2 深度学习

![image-20240630205924404](./assets/image-20240630205924404.png)

对于transformer模型架构，实现其相关的所有layer，并调整参数训练续写任务

>在Attention is all you need 一文中，transformer结构如下
>
>![image-20240630213149957](./assets/image-20240630213149957.png)
>
>encoder将符号表示的输入序列 $(x_1,...,x_n)$映射成一个连续表示的序列 $z = (z_1,...,z_n)$
>
>给定z，decoder将一次生成一个字符，逐渐将所有的输出序列 $(y_1,...,y_m)$生成，每次在生成下一个字符时，将之前生成的符号作为附加输入

### 实现

首先要实现相关的所有layer和model，这里只减少一些关键的部分，相对冗余的代码填空不做说明

#### tokenization

对于词元化任务，我选择 `单词级` 划分 `vocabulary`

并对 开始符，终结符和 未知符分别用 0, 1, 2标记，注意在decode阶段取消终结符，以免生成到一个终结符就结束译码

```python
def generate_vocabulary(
        self,
        ):
        self.word2index["<sos>"] = 0
        self.word2index["<eos>"] = 1
        self.index2word[0] = "<sos>"
        self.index2word[1] = "<eos>"
        # unk字符
        self.word2index["<unk>"] = 2
        self.index2word[2] = "<unk>"
        # enc_output = self.enc.encode(self.dataset)
        index = 2
        # 不区分大小写
        dataset_lowercase = self.dataset.lower()
        enc_output = self.enc.encode(dataset_lowercase)
        for token in enc_output:
            if self.token_str(token) not in self.word2index:
                index += 1
                self.word2index[self.token_str(token)] = index
                self.index2word[index] = self.token_str(token)
        self.vocab_size = len(self.word2index)
        print("Vocabulary size: ", self.vocab_size)
```

##### 编码和解码

`encode`

```python
def encode(
    self,
    sentence : str,
) -> torch.Tensor:
    sentence = sentence.lower()
    sentence = self.enc.encode(sentence)
    for i in range(len(sentence)):
        if self.token_str(sentence[i]) not in self.word2index:
            sentence[i] = "<unk>"
        else:
            sentence[i] = self.token_str(sentence[i])
            return torch.tensor([self.word2index["<sos>"]] + [self.word2index[token] for token in sentence], dtype=torch.long)
```

`decode`

```python
def decode(
    self,
    tokens : torch.Tensor,
) -> str:
    ans=""
    for i in tokens[1:-1]:
        if(i==1):
            break
            if(i==0):
                continue
                if(i==2):
                    continue
                    ans+=self.index2word[i].decode("utf-8")
                    return ans
```

#### attention机制

注意力的计算公式为
$$
Head = Attention(x) = Softmax(M \cdot QK^T)V\\
Q = xW_q, K = xW_k, V = xW_v
$$
点击注意力机制主要计算查询向量和key之间的点积，然后将结果进行归一化得到注意力权重

![image-20240630212810737](./assets/image-20240630212810737.png)



```python
import math
def attention(q, k, v, mask=None, dropout=None):
    "Compute ''Scaled Dot Product Attention'"
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.to(device)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn
```

- 核心是计算Q,K,V三个矩阵，同时需要根据注意力分数和掩码选择要考虑某几位
- 这还定义了dropout机制用于调参

#### 多头注意力机制

![image-20240630214004643](./assets/image-20240630214004643.png)

即多头注意力机制如下

>![image-20240630214053448](./assets/image-20240630214053448.png)

我们要在这个class内实现这些layer

```python
def __init__(self, n_heads:int, head_size:int, seq_len:int, embed_size:int):
        # n_heads is the number of head attention
        # head_size is the hidden_size in each HeadAttention
        super().__init__()
        head_size = embed_size // n_heads
        #TODO: implement heads and projection
        self.heads = nn.ModuleList(
            [HeadAttention(seq_len, embed_size, head_size) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(embed_size,n_heads * head_size)
        nn.init.xavier_uniform_(self.projection.weight)
```

前向传播计算如下

```python
def forward(self, inputs):
        output = []
        inputs = inputs.to(device)
        for h in self.heads:
            temp = h(inputs)
            output.append(temp)
        output = torch.cat(output, dim=-1)
        output = self.projection(output)
        return output
```

#### 专家网络Expert

呃，从我学习transformer的路径来看，这其实是 FeedForward前馈网络的内容，实际上就是一个两层线性层的mlp，考虑注意力机制可能对复杂过程的拟合程度不够，通过增加两层网络来增加模型的能力，实现一种语义汇聚

把同一个mlp对每个输入序列中的token作用一次
$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

- 先对输入进行一个线性变换
- 通过非线性激活函数ReLU
- 第二次线性变换

模型定义如下

```python
def __init__(self, embed_size:int, dropout=0.1):
        super().__init__()
        #TODO: init two linear layer
        d_model = embed_size
        d_ff = embed_size * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
```

前向传播计算如下

```python
def forward(self, inputs):
        inputs = inputs.to(device)
        outputs = self.w_2(self.dropout(F.relu(self.w_1(inputs))))
        return outputs
```

#### 选通网络 TopkRouter

选通网络决策每个embedding层要使用的epxert计算

具体计算方式如下

![image-20240630215008449](./assets/image-20240630215008449.png)

模型定义如下

```python
def __init__(self, embed_size, num_expert, active_experts):
        super().__init__()
        self.embed_size = embed_size
        self.num_expert = num_expert
        self.active_experts = active_experts
        to_experts = nn.Linear(embed_size, num_expert)
        self.noise = nn.Linear(embed_size, num_expert)
        nn.init.xavier_uniform_(to_experts.weight)
        self.to_experts = to_experts
```

前向传播计算如下

```python
def forward(self, inputs):
        score = self.to_experts(inputs)
        rand = torch.rand_like(score)
        noise = F.softplus(self.noise(inputs))
        score = score + noise * rand
        
        score = torch.sum(score, dim=1)
        score = torch.sum(score, dim=0)

        score /= (inputs.size(0) * inputs.size(1))

        _, idx = torch.topk(score, self.active_experts, dim=-1)
        indices = torch.zeros_like(score)
        indices.scatter_(-1, idx, 1)

        score = score.masked_fill(indices == 0, float("-inf"))
        router_output = F.softmax(score, dim=-1)

        return router_output, indices
```

#### 稀疏专家网络 SparseMoE

![image-20240630215132557](./assets/image-20240630215132557.png)

对于稀疏专家网络，定义相关的model和前向传播计算的过程

这里几乎是做代码的复现，因此不展示代码实现

### 训练

在training部分，我定义了一些超参数，并设置了两种 优化器 （`SGD`与 `Adam`），以下是一些超参的设置

#### 超参数（hyper-parameter）

```python
EBD_SIZE, TRANS_N, EPOCH_NUM = 64, 4, 40
learning_rate = 1e-3
weight_decay = 1e-5
lr_period = 10
lr_decay = 0.1
momentum = 0.9
optimizer_func = 'Adam'
```

- `EBD_SIZE`是 `embed_size`的大小，
- `TRANS_N`是transformer层的个数，5层效果会明显比4层好一点
- `EPOCH_NUM`是迭代训练的周期数，这里设置40个epoch在 kaggle提供的P100上需要训练60分钟左右，而在我本地的3050上的循环周期实在是太长了，因此后续调参在其大概收敛到loss < 0.1后我就没有再加epoch了
- `learning_rate`是学习率，直接固定学习率会导致epoch最后几轮的时候更新不明显，这可能是由于学习率相对于梯度变化量太大，导致步长太大，难以有效更新，所以后续采用了scheduler动态调整学习率
- `lr_period`和 `lr_decay`是动态调整学习率的两个超参数，分别对应 调整学习率的周期和每次调整的比例，这里我选择每10个epoch将`learning_rate`更新为原先的0.1
- `momentum`是冲量，主要在 `SGD`（随机梯度下降）中有所作用，但经过我几次训练测试发现，`Adam`方法效果明显优于 `SGD`，因此 `momentum`后续没什么作用了
- 这里我选择的 优化器是 `Adam` 效果明显优于 `SGD`

#### 模型

模型例化定义

```python
model = SparseMoETransformer(
    vocab_size=len(tokenizer.word2index),
    seq_len=50,
    embed_size=EBD_SIZE, 
    n_layers=TRANS_N, 
    n_heads=8, 
    num_expert=8, 
    active_experts=2
).to(device)
model = model.to(device)
```

#### 训练环境

对于超过2层的transformer结构，在cpu上训练动辄20个小时，调参成本太大，而且训练效果很有限，因此我选择在了两种训练环境

- 本地 RTX 3050 conda 12.4
- kaggle 上 每周 30h free的 P100

在前者上的训练效果也比较有限，我并没有调出效果很好的模型

在后者我在一组超参数下 大概得到了如下的训练结果，我将在 **测试结果** 展示

### 测试结果

在这组超参数和模型下

```python
EBD_SIZE, TRANS_N, EPOCH_NUM = 64, 6, 4
learning_rate = 1e-3
weight_decay = 1e-5
lr_period = 10
lr_decay = 0.1
momentum = 0.9
optimizer_func = 'Adam'
model = SparseMoETransformer(
    vocab_size=len(tokenizer.word2index),
    seq_len=50,
    embed_size=EBD_SIZE, 
    n_layers=TRANS_N, 
    n_heads=8, 
    num_expert=8, 
    active_experts=2
).to(device)
```

我训练 4个 `epoch`得到了如下的训练结果

采用对数坐标系（普通坐标系loss后几轮loss变换实在太小了），结果如下

![image-20240701160837508](./assets/image-20240701160837508.png)

续写结果如下

![image-20240701160820748](./assets/image-20240701160820748.png)

## 2.3 对语言模型进行对抗攻击

bonus 实验：利用GCG算法攻击 语言模型，得到任意输出

### background info

#### 越狱（jailbreaking）

指通过精心设计输入给语言模型的prompt，使其绕过原有的限制，产生并非设计初衷的输出内容。

这种现象说明语言模型的输出高度依赖于输入的prompt，并且通过调整prompt，我们可以显著影响生成的内容

![image-20240701133747077](./assets/image-20240701133747077.png)

#### GCG算法

GCG 的优化目标是最大化模型输出的第一段内容是"确认性信息"的概率

![image-20240701134325376](./assets/image-20240701134325376.png)

### 代码实现

`token_gradients`函数

```python
# TODO: input_ids 是整个输入的 token_id, 但是我们只需要计算 input_slice 的梯度
    # 1. 先定义一个 zero tensor，shape 为 (input_slice_len, vocab_size)
    # vocab_size 是词表大小，思考词表大小对应模型的什么矩阵的哪一维
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0], 
        embed_weights.shape[0], 
        device=device, 
        dtype=embed_weights.dtype
    ) 
    # TODO: 2. 将 one_hot 中对应的 token_id 的位置置为 1
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1), 
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    # TODO: 3. 将 one_hot 乘以 embedding 矩阵，得到 input_slice 的 embedding，注意我们需要梯度
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    #TODO: 4. 用 input_embeds 替换 embedding 的对应部分（可以拼接），拿到 logits 之后和 target 进行 loss 计算
    full_embeds = torch.cat(
        [input_embeds, embeds[:, input_slice.stop:, :]],
        dim=1
    )
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
```

`sample_control`函数

```python
control_toks = control_toks.to(grad.device)
    # TODO: 重复 batch_size 次（随机采样的次数） -> (batch_size, len(control_toks))
    original_control_toks = control_toks.repeat(batch_size, 1)

    # TODO: 生成 batch_size 个新的 token 位置作为采样的位置，允许复选
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device = grad.device
    )
    new_token_pos = new_token_pos.type(torch.int64)

    # TODO: 利用梯度的 topk 来获取每个 token 位置上梯度最大的 topk 个 token 的索引
    # https://pytorch.org/docs/stable/generated/torch.topk.html
    top_indices = (-grad).topk(topk, dim=1).indices

    # TODO: 从 top_indices 中的 new_token_pos （作为 index）随机采样一个 topk token 的索引，作为新的 token
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device)
    )
    
    # TODO: 得到新的 control tokens
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )
```

`is_success`函数

```python
# TODO: 这部分需要你实现判断是否攻击成功的逻辑。用你的 prompt 进行一次推理，
    # 然后判断输出中是否包含 target，如果包含则返回 True，否则返回 False
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids.to(device),
        max_new_tokens=50,
        num_beams=1,
        temperature=0
    )
    output = output[:, len(input_ids[0]):]
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

画图部分

```python
#TODO: 定义 adv_slice, target_slice, loss_slice
adv_slice = slice(0, len(tokenizer.encode(
    adv_string_init, add_special_tokens=False
)))
target_slice = slice(adv_slice.stop, adv_slice.stop + len(tokenizer.encode(target, add_special_tokens=False)))
loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
```



### 测试结果

大概经过70个epoch，当 `loss`接近 0.552时，越狱成功，运行结果如下

![image-20240701154435709](./assets/image-20240701154435709.png)

![image-20240701154445227](./assets/image-20240701154445227.png)