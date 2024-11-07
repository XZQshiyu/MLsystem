### 一个Dialect的结构

- 基本信息：

  - name：方言的名称
  - summary：对方言的简短总结，通常为一句话描述其目的
  - description：方言的详细描述，包括其功能和应用场景

- 依赖关系：

  dependentDialects：该方言在构造时需要加载的其他方言的列表。

- 命名空间：

  - cppNamespace：C++命名空间，用于存放该方言的操作

- 额外声明：

  - extraClassDeclaration：可选的代码块，用于在方言声明中放置额外的声明

- 功能钩子

  **hasConstantMaterializer**: 如果方言覆盖了物化常量的钩子，设置为 1。

  **hasNonDefaultDestructor**: 指示方言是否提供了非默认析构函数的实现。

  **hasOperationAttrVerify**: 如果方言覆盖了验证操作属性的钩子。

  **hasRegionArgAttrVerify**: 如果方言覆盖了验证区域参数属性的钩子。

  **hasRegionResultAttrVerify**: 如果方言覆盖了验证区域结果属性的钩子。

  **hasOperationInterfaceFallback**: 如果方言覆盖了操作接口的回退钩子。

  **hasCanonicalizer**: 如果方言覆盖了规范化模式的钩子。

  **isExtensible**: 指示该方言是否可以在运行时扩展，添加新的操作或类型。

- 属性与类型解析

  **useDefaultAttributePrinterParser**: 如果设置为 1，ODS（Operation Definition Schemas）会为属性解析和打印生成默认的实现。

  **useDefaultTypePrinterParser**: 如果设置为 1，ODS 会为类型解析和打印生成默认实现。

- 属性存储：

  **usePropertiesForAttributes**: 指定是否将 ODS 定义的固有属性存储为属性，默认为 1