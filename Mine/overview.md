# My Learning Strategy for ML System

我觉得我需要有一份完整的技术路线，告诉我我目前需要学些什么，目前的研究热点是什么，以便在一个 繁而杂的方向中找到出路。

学习形式无所谓，无论是paper，survey，还是course，lecture，lab，甚至是tutorial，documentation，都可以，只要能满足我对这个技术的需求就可以，以下是我的总结。

## 基础知识

### 数学

#### 线性代数

MIT线性代数

https://www.bilibili.com/video/BV16Z4y1U7oU/?spm_id_from=333.337.search-card.all.click&vd_source=11b96aa488d3f3b390985d70e5eedc4d

#### 概率论与数理统计

UCB CS126

https://csdiy.wiki/%E6%95%B0%E5%AD%A6%E8%BF%9B%E9%98%B6/CS126/

#### 机器学习

以吴恩达老师的机器学习课程 及 分布式机器学习一书为主。不关注代码实现，关注理论和数学推导

### system

#### operating system

- MIT 6.S081
- jyy老师操作系统
- 阅读ostep 

#### distributed system

- MIT 6.824
- THU陈康老师分布式系统

#### compiler

呃，感觉是最不需要补的一块，CS143吧

#### network

- CS144

#### database

- CMU 15-445 

### parallel computing

#### parallel computing basics

- #### CMU15418 Parallel computing

#### HPC

- #### UCB cs267 Applications of Parallel Computers

#### cuda

cuda生态链学习

### AI

#### deep learning

- UMich EECS 498-007
- 李宏毅老师机器学习 

#### LLM inference



#### MLsys

- 中科大 智能计算系统基础

## 技术栈

### 编程语言 & tools

#### C++

更重要的是modern C++，C++11及之后的特性，对C++手册检索足够熟悉

#### Python

不只是停留在基础的python代码，更多如cpython等，高级用法和特性

#### CMake

#### git

#### docker

#### linux，ssh，shell

### cuda & DSA

我认为足够了解cuda，gpu及相关DSA结构是至关重要的，无论是华为的昇腾NPU和CCE，还是诸多DSA，AI加速器，体系结构上的了解都是必不可少的

### AI 编程框架

#### pytorch

对pytorch的基本搭建神经网络，多卡训练，等方法足够了解，和一些pytorch固有算子及特性

#### framework

目前粗略的了解就是DeepSpeed等等了

### AI Compiler

#### mlir(llvm-project)

triton等工具都是基于mlir实现，对mlir的dialect，conversion编写，及算子lowering流程至关重要

#### triton

triton已经逐渐成为一种pytorch编译流程中的一种方案了，基于mlir实现，也需要对triton有足够的了解

#### tvm

与mlir相对的一种AI 编译器的框架



## paper & survey

主要值得关注的方向有

- LLM inference：
  - 场景：MoE，RAG，多模态（如Sora）
  - 手段：
    - KV-cache
    - prefill & decode
    - 并行策略
    - 优化手段
  - 通信问题
- AI compiler：

目前先以综述为主，然后关注于最早的模型本身，LLM推理的全过程，再到不同场景的综述，再到具体的手段。

