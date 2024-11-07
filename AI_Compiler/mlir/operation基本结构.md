```c++
// Base class for all ops.
class Op<Dialect dialect, string mnemonic, list<Trait> props = []> {
  // The dialect of the op.
  Dialect opDialect = dialect;

  // The mnemonic of the op.
  string opName = mnemonic;

  // The C++ namespace to use for this op.
  string cppNamespace = dialect.cppNamespace;

  // One-line human-readable description of what the op does.
  string summary = "";

  // Additional, longer human-readable description of what the op does.
  string description = "";

  // Optional. The group of ops this op is part of.
  OpDocGroup opDocGroup = ?;

  // Dag containing the arguments of the op. Default to 0 arguments.
  dag arguments = (ins);

  // The list of results of the op. Default to 0 results.
  dag results = (outs);

  // The list of regions of the op. Default to 0 regions.
  dag regions = (region);

  // The list of successors of the op. Default to 0 successors.
  dag successors = (successor);
`
  list<OpBuilder> builders = ?;

  bit skipDefaultBuilders = 0;

  string assemblyFormat = ?;

  bit hasCustomAssemblyFormat = 0;

  bit hasVerifier = 0;

  bit hasRegionVerifier = 0;

  bit hasCanonicalizer = 0;

  // rewrite patterns".
  bit hasCanonicalizeMethod = 0;

  bit hasFolder = 0;

  bit useCustomPropertiesEncoding = 0;

  // Op traits.
  // Note: The list of traits will be uniqued by ODS.
  list<Trait> traits = props;

  code extraClassDeclaration = ?;

  code extraClassDefinition = ?;
}
```

传入的参数

- Diaclet dialect：这定了操作所属的方言，方言是MLIR种的一种扩展机制，用于定义特定的操作和类型。每个操作必须归属于某个方言，确保操作的上下文和语义清晰
- string mnemonic：操作的助记符，用于唯一标识该操作。助记符通常是一个简短的名称，方便在 MLIR代码中引用该操作。
- `list<Trait> props = []`：这个可选参数是一个特性列表，定义了操作的行为或约束，特性可以包括
  - 是否具有可变参数的支持
  - 是否支持特定的优化或转换
  - 操作的区域和结果的性质等





- 基本信息：

  - opDialect：操作所属的方言，类型为 `Dialect`
  - opName：操作的助记符，用于标识操作
  - cppNamespace：指定用于该操作的C++命名空间，默认为所属方言的命名空间

- 描述信息

  **summary**: 操作的一行简要描述。

  **description**: 操作的详细描述，通常包含用法示例。

- 文档与参数

  **opDocGroup**: 可选，指示操作所属的文档组。

  **arguments**: DAG（有向无环图）格式的输入参数，默认无参数。

  **results**: DAG 格式的输出结果，默认无结果。

  **regions**: DAG 格式的区域，默认无区域。

  **successors**: DAG 格式的后继操作，默认无后继。

- 构建器与格式

  **builders**: 自定义构建器列表，生成操作实例的代码。

  **skipDefaultBuilders**: 如果设置为 1，避免生成默认构建函数，必须提供自定义构建器。

  **assemblyFormat**: 可选，定义操作的自定义装配格式。

  **hasCustomAssemblyFormat**: 如果设置为 1，表明操作有自定义的装配格式，支持自定义解析和打印方法。

- 验证与优化

  **hasVerifier**: 如果设置为 1，表明该操作有额外的验证逻辑，生成 `verify()` 方法。

  **hasRegionVerifier**: 如果设置为 1，生成 `verifyRegions()` 方法，验证与区域相关的额外不变式。

  **hasCanonicalizer**: 指示该操作是否有关联的规范化模式。

  **hasCanonicalizeMethod**: 如果设置为 1，指示操作具有静态的规范化方法。

  **hasFolder**: 如果设置为 1，表明该操作具有折叠功能。

- 属性与编码

  **useCustomPropertiesEncoding**: 如果设置为 1，允许操作实现自定义的 `readProperties` 和 `writeProperties` 方法以发出字节码。

  **traits**: 操作的特性列表，可能包括不同的行为和约束。

- 额外代码

  **extraClassDeclaration**: 可选，额外声明代码，添加到生成的操作类中。

  **extraClassDefinition**: 可选，额外代码，添加到生成的源文件中。