# Operation Definition Specification(ODS)

MLIR supports defining **operations and data types** in a table-driven manner

- via TableGen

  a generic language and its tooling to maintain records of domain-specific information.

- an operation will be expanded into an equivalent `mlir::Op` C++ template specialization at compiler build time.

## Motivation

每个 dialect 包含了一系列的operations。MLIR允许用户定义自己的方言，包含一系列自定义的操作，虽然这样很灵活，但也引发了 stringly 类型 IR的一些问题

- 需要大量的字符串比较
- 操作的方式不直观，`getOperand(3)`这种接口不够直观，需要用户去判断operand3是什么
- 操作的定义较为通用，返回类型比较宽泛，可能会导致类型不匹配问题
- IR文本过于冗长

verification的问题

- best case：维护通过一个字符串到verification function的映射表，但仍有字符串匹配
- medium case：在整个代码库中可能存在重复，难以维护
- worst case：没有验证函数，可能导致operation在无约束的情况下不当使用

solution：基于 tablegen-driven的定义

- 对于每个dialect，有一个集中的位置，包含每个operation的所有信息，包括约束条件和自定义汇编格式
- 方便build、verifiy、parse、print、analysis，减少冗余代码

## Benefit

与C++ template的定义比较

- single source of truth

  所有信息都集中在一个 record中，定义这个operation的所有内容

- removing boilerplate

  根据record自动生成许多常用的辅助方法，例如操作数、属性结果的获取方法，操作的构建方法、验证方法等

- facilitating auto-generation

  不仅用于操作定义本身，还可以驱动其他组件的自动生成：如计算图

## TableGen Syntax

MLIR中用于指定操作信息的语言 `TableGen`：提供了一种 record 的语法，用于定义各种操作及其相关信息。通常，这些定义文件的后缀是 `.td`

- `class`：类似C++类，可以被模版化和继承（subclassed）。这允许定义具有不同特征的类，并在后续进行扩展和复用

- TableGen 定义（def）：

  类似于 C++ 中的对象。可以通过 特化（specializing）一个 TableGen 类来声明，也可以完全独立声明

  ```c++
  def MyDef : MyClass<...>;
  def MyDef;
  ```

- 有向无环图（dag）：

  ableGen 中的 **dag** 是一个专门用于表示元素的有向无环图的类型。一个 **dag** 有一个操作符（operator）和零个或多个参数（arguments）。其语法格式为 `(operator arg0, arg1, argN)`。

  操作符可以是任何 TableGen 定义（def）；参数可以是任何类型，包括另一个 **dag**。同时，可以为操作符和参数附加名称，例如 `(MyOp:$op_name MyArg:$arg_name)`。

## Operation Definition

MLIR defines sereral common constructs to help operation definition and provide their semantics via a special `TableGen backend`