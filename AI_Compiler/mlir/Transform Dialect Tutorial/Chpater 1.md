## Chpater 1：Combining Existing Transformations

### Introduction

The Transform dialect allows one to precisely target transformations at specific operations in the IR and to chain them, that is to apply a transformation to operations produced by the previous transformation.

trasformations are expressed as other operations in the IR.

两种 IR 

- **transform IR**：包含 这些 operations 的IR
- **payload IR**：被转换的 IR

**Transform IR**

Transform IR operations operate on values that may be associated with payload IR operations, values or attributes.

-  payload IR operations : operatoin
- payload IR value : value handles
- payload IR attributes : parameters

变换IR的应用总是从一个顶层操作开始，在 C++ API 中，这个 operation 被传递给了 `applyTransfroms`函数。

- 这个operation 指定是否应该执行其他转换，以及如何执行
- 最常见的 top-level operation， `transform.named_sequence`  仅仅应用其主体中列出的一个接一个的其他转换操作，类似于 函数 或 宏

用一个简单的 transformations 的 sequence 基于 `fully connected + bias + ReLU`

>这个算法可以归结为执行一个矩阵乘法，然后是一个（elementwise的）矩阵加法，最后得到一个 elementwise的最大值为0，这可以用如下的IR表示

```c++
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
```

### Top-Level Sequence Operation

处于性能考虑，我们将对这些operation做 **tiling** 和 **fusing**去发掘cache locality

```c++
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">):
    transform.yield
  }
}
```

>`transform.named_sequence`是 `Transform Dialect`中的一个功能，它允许用户定义一个变换序列，并为其命名。
>
>命名的序列意味着其他地方可以通过名字调用这个序列来复用其中的变换逻辑
>
>通过定义为 `named_sequence`，你可以更清晰地将特定的变换操作封装成一个独立的模块，并在需要时调用。
>
>这些是传递给 `named_sequence` 的参数：
>
>- `%arg0` 的类型是 `!transform.any_op`，表示这个参数可以是任何操作。
>- `%arg1` 的类型是 `!transform.op<"linalg.matmul">`，表示这个参数专门匹配 `linalg.matmul` 操作。
>- `%arg2` 的类型是 `!transform.op<"linalg.elemwise_binary">`，表示这个参数匹配 `linalg.elemwise_binary` 操作。
>
>在 `Transform Dialect` 中，通过这种方式可以为变换序列传递操作符，并基于这些操作符进行各种优化和转换操作
>
>`transform.yield` 用于终止 `named_sequence`。在这个特定的例子中，它表明该序列不执行任何实际操作。
>
>通常情况下，在 `named_sequence` 中你会看到各种变换操作，而不是直接 yield。在这里，可能是一个定义了参数的空模板

通过 `applyTransforms` 或者 `applyNameSequence`

剩下的 entry block arguments是可选的，并与 待转换IR的 attributes，operation和values有关

### Failure Propagation

Transform dialect拥有一个独特的处理 diagnostics，支持可恢复错误的机制。

failure propagation mode有两个 选项

- propagate：如果嵌套转换中的任何一个失败，则使序列转换失败
- suppress：即使其中一个嵌套转换失败，也会使序列成功，但不会尝试在序列中失败的转换之后执行转换

要检查或调试转换序列，可以打印与转换 IR 值相关联的各种实体。例如，我们可以打印与句柄关联的操作

```c++
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  transform.debug.emit_remark_at %arg1, "matmul"
      : !transform.op<"linalg.matmul">
  transform.debug.emit_remark_at %arg2, "elemwise_binaries"
      : !transform.op<"linalg.elemwise_binary">
  transform.yield
}
```

### Transform Dialect Interpreter

我们并不想在每次更改转换时都重新编译编译器，所以我们可以使用 Transform dialect interpreter pass 将这个转换序列应用于有效负载 IR

```bash
$ mlir-opt sequence.mlir --pass-pipeline="
    builtin.module(transform-interpreter{
        debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})"
```

### Specifying Transformations

tiling the matmul operation

```c++
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // The actual tiling transformation takes tile sizes as attributes.
    %loop, %tiled = transform.structured.tile_using_forall %arg1
                    tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">)
     -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

运行完结果为

```c++
func.func @fc_relu(%arg0: tensor<512x512xf32>,
                   %arg1: tensor<512x512xf32>,
                   %arg2: tensor<512x512xf32>,
                   %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.forall (%arg4, %arg5) in (128, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
    %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
    %4 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg5)
    %extracted_slice = tensor.extract_slice %arg0[%3, 0] [4, 512] [1, 1]
                     : tensor<512x512xf32> to tensor<4x512xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [512, 32] [1, 1]
                       : tensor<512x512xf32> to tensor<512x32xf32>
    %extracted_slice_1 = tensor.extract_slice %arg6[%3, %4] [4, 32] [1, 1]
                      : tensor<512x512xf32> to tensor<4x32xf32>
    %5 = linalg.matmul
         ins(%extracted_slice, %extracted_slice_0
             : tensor<4x512xf32>, tensor<512x32xf32>)
         outs(%extracted_slice_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg6[%3, %4] [4, 32] [1, 1]
          : tensor<4x32xf32> into tensor<512x512xf32>
    }
  }
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
    ins(%1, %cst : tensor<512x512xf32>, f32)
    outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %2 : tensor<512x512xf32>
}
```

### Handle Invalidation and Expensive Checks Mode