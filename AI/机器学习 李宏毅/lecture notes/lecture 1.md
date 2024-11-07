## Lecture 1 Introduction of Deep Learning

### Machine Learning overview

实际上找到一个函数（function）实现一些功能，提供输入，得到理想的输出

如

- Speech Recognition 

  ![image-20240124172023302](./assets/image-20240124172023302.png)

- Image Recognition

  ![image-20240124172031292](./assets/image-20240124172031292.png)

- Playing Go：预测 next move

  ![image-20240124172036817](./assets/image-20240124172036817.png)

#### Different types of Functions

Regression：The function outputs a scalar（一个数值）

- 从输入的参数得到一个具体的数值

![image-20240124172149919](./assets/image-20240124172149919.png)

Classification：Given options(classes), the function outputs the correct one.

- 对一些集合中的选择做出一个选择

![image-20240124172254410](./assets/image-20240124172254410.png)

>比如AlphaGo是一个Classification 问题，输入的一个棋盘的当前位置，可选的输出集合是棋盘19$\times$19上所有的位置，选择一个位置作为输出结果（next move）
>
>![image-20240124172410873](./assets/image-20240124172410873.png)

Structured Learning：Create something with structure(image, document)

- 让机器学会创造这件事

### How to find a funciton？

三个步骤

#### 1.Function **with Unnknown Parameters**

基于以前的信息（domain knowledge）提出一个预测函数，并规定一个b和w的初始值

如

$y = b + w x_1$

- $x_1$是特征值，也即input
- w和b是**未知参数**，初始参数是手动设定的，是一个hyper parameter，需要通过训练集得到
  - w即weight：对于输入特征$x_i$其对于预测结果的影响占所有输入特征的权重
  - b即bias，预测值相对于初始值的偏置量
- y是预测的结果，与实际结果比较

>下面是一个对于一个具体问题的预测函数（带未知数的）
>
>![image-20240124173447450](./assets/image-20240124173447450.png)

#### 2.Define Loss from Training Data

Loss is a function of parameters

- 比如$L(b,w)$是基于参数bias和weight的loss函数
- Loss 是量度预测结果好坏的函数（通过比较$\hat{y}$与y，实际上也是在判断未知参数的好坏）

>比如，对于先前的问题，一个loss function在做的比如是将预测的1.2的播放量与1.2世纪的播放量比较
>
>![image-20240124174112868](./assets/image-20240124174112868.png)

两种常见的Loss函数的定义

Loss：$L = \frac{1}{N}\sum_{n}^{N}e_n$

- N为参与预测的个数

MAE：L is mean absolute error

$e = |y - \hat{y}|$

MSE：L is mean square error

$e = (y - \hat{y})^2$

如果$y$和$\hat{y}$都是概率分布的话，采用的方法为 `Cross-entropy`（交叉信息熵的方式）

>一个真实的Loss函数的案例，Small L代表未知参数的选择比较好，使得模型预测更精准一些，而Large L代表比较差
>
>![image-20240124174457646](./assets/image-20240124174457646.png)

#### 3.Optimization

$w^*,b^* = arg~min_{w,b}~L$

- 表示从所有的w和b中找到一组使得Loss函数最小的w和b，也即最优结果$w^*,b^*$

采用的方式为**梯度下降（Gradient Descent）**

比如计算w的最优结果$w^*$

- 随机选取一个起始点$w^0$
- 不断计算L相对于w的偏导，$\frac{\partial L}{\partial w}|_{w = w^0}$
- 更新w, $w^i = w^{i-1} - \eta \frac{\partial L}{\partial w}|_{w = w^{i-1}}$
  - 如果偏导为正，则相当于降低w，最小值出现在左侧
  - 如果偏导为负，则相当于增加w，最小值出现在右侧
  - $\eta$为学习率（learning rate），是一个hyperparameters（即自己设定的），相当于梯度下降的步长，即发现不是最优的时候左右移动的距离
- 迭代更新w，直到设定的更新次数结果或者找到一个偏导为0的w

>![image-20240124175014895](./assets/image-20240124175014895.png)

梯度下降的一个潜在问题

- 很可能发生只找到一个Local minima而不是global minima的解，即局部最优而不是全局最优的解

多个参数的gradient descent的拓展

![image-20240124175645720](./assets/image-20240124175645720.png)

>一个梯度下降的实例
>
>![image-20240124175712498](./assets/image-20240124175712498.png)

#### Training

上述的过程都是训练的过程，即基于原有的数据集从某个起点开始通过对原有数据集进行预测和更新，得到一个函数，但并没有用来预测未知的数据集

- 如上述训练过程并没有预测还没有发生的日期的播放量

训练的过程可以分为上面所说的三个步骤

- function with unknown
- define loss from trainning data
- optimization

通过预测的结果在真实值上运行的结果，可以修改模型（即修改预测函数），使得预测的结果更加精准

##### Linear models

- 比如在上述训练过程，对下一天播放量的结果预测可以不只是基于昨天，而是基于过去7天的，即有7个feature$x_i$，或者更多的feature

>![image-20240124180125180](./assets/image-20240124180125180.png)

上述的模型都是**线性模型（Linear models）**

对于线性模型 $y = wx + b$而言

- w代表直线斜率
- b代表截距
- 无论如何修改w和b都只能得到一个单调的预测函数，这并不精确

![image-20240124180343726](./assets/image-20240124180343726.png)

这是Linear model 的 **Model Bia**s 即线性模型本身的**严重限制(severe limitation)**

##### Piecewise Linear Curves

![image-20240124180809680](./assets/image-20240124180809680.png)

任何一条上图红色的折线，都可以通过一系列piecewise linear（上图蓝色的线）组合得到，

所谓piecewise linear就是在某一特定段的线性函数，其余部分是平行于x轴的，组合这些piecewise linear并加上一个常数就可以得到任意一条折线

![image-20240124180650985](./assets/image-20240124180650985.png)

对于任意一条曲线，都可以用足够多的piecewise linear组合起来得到，因此只需要表示出每一条piecewise linear就可以

表示piecewise linear的方式有两种

- sigmoid function
- ReLU

###### sigmoid function

即通过一个函数

$y = c\frac{1}{1 + e^{-(b+wx_1)}}$

其中$\frac{1}{1+e^{-r}}$这一部分就是所谓的**sigmoid(r)**

在特征值x很小的时候sigmoid function近似趋于0，x很大的时候sigmoid function近似趋于c，可以近似一条piecewise linear

>![image-20240124181223032](./assets/image-20240124181223032.png)

由此可以得到新的模型函数（step 1 中的function with unknown parameters）

![image-20240124181321290](./assets/image-20240124181321290.png)

其他的两种表示方式

![image-20240124181355376](./assets/image-20240124181355376.png)

![image-20240124181433446](./assets/image-20240124181433446.png)

- x为feature
- unknown：把这些向量都取出来得到了向量$\theta$，即所有的未知参数
  - W
  - b
  - $c^T$
  - b

<img src="./assets/image-20240124181552576.png" alt="image-20240124181552576" style="zoom:50%;" />

###### ReLU

![image-20240124182313474](./assets/image-20240124182313474.png)

两个ReLU可以得到一个hard sigmoid

![image-20240124182341977](./assets/image-20240124182341977.png)

### 重新定义training 的步骤

#### 1.Functin with Unknown parameters

![image-20240124181713943](./assets/image-20240124181713943.png)

![image-20240124181723596](./assets/image-20240124181723596.png)

#### 2.Define Loss from Training Data

![image-20240124181750423](./assets/image-20240124181750423.png)

#### 3.Optimization

![image-20240124181829265](./assets/image-20240124181829265.png)

迭代计算$\theta$ 及其梯度并进行梯度下降更新，直到某次梯度计算为0或者到达迭代次数达到我们预期的次数后，训练结束

对于具体的训练过程，我们会把拥有N个特征的数据集分割成多个batch，每一轮训练称为一个epoch，每个epoch每次取出一个batch进行梯度下降计算，然后再取下一个batch计算，直到所有的batch都用于更新了一遍参数$\theta$

- 因此模型训练的次数为epoch
- 每个epoch内模型参数update的次数为batch的个数
- 所有的batch构成了训练集

![image-20240124182228874](./assets/image-20240124182228874.png)

>example
>
>![image-20240124182249230](./assets/image-20240124182249230.png)

##### activation function

![image-20240124182408419](./assets/image-20240124182408419.png)

### 多layers的模型

![image-20240124182553706](./assets/image-20240124182553706.png)

- 用第一次optimization的结果a用于下一次opmization计算，注意在不同层之间传递a的时候并没有对w和b进行梯度下降更新，但是换用了一批新的b和w进行模型预测计算
- 有多少层就是进行了多少次没有经过梯度下降的模型预测

#### Deep Learning

![image-20240124182836875](./assets/image-20240124182836875.png)

![image-20240124182852650](./assets/image-20240124182852650.png)

Overfitting：Better on training data，worse on unseen data

