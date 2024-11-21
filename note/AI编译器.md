# AI编译器



## AI编译器

### 依赖的层级关系

在下面的图表中，左侧展示了传统编译器的软件栈结构，右侧则呈现了AI编译器的架构。通过对比这两部分，我们可以清晰地辨识出它们之间的差异。

传统编译器的前端专注于对高级编程语言进行深入的语义分析、语法分析和词法分析，将源代码转化为中间表示（IR）。在中间阶段，编译器执行一系列优化Pass，专门针对高级语言代码进行性能提升。而在后端，编译器负责处理代码的具体布局、寄存器分配等任务。

相比之下，AI编译器的架构则有显著的不同。它的前端主要负责将深度神经网络的API表达为计算图，这一过程涉及到模型的构建和转换。在中间优化阶段，AI编译器专注于图算融合、算子融合、自动微分和并行切分等特定优化技术。后端则根据目标硬件平台，对kernel进行定制化优化，确保代码在不同硬件上都能高效运行。在某些情况下，如CPU或TPU等芯片，AI编译器甚至可能利用类似LLVM这样的传统编译器技术。

由此可见，AI编译器在很多方面是站在传统编译器的肩膀上，它们之间形成了一种互补和协同的关系。

![img](https://chenzomi12.github.io/_images/01Appear11.png)









![img](https://chenzomi12.github.io/_images/stage.png)







以深度学习中一个常见的MatMul+Add+Relu计算图为例，看一下TVM做代码生成的一个过程。首先TVM将接受的计算图转换为TVM中的领域特定语言Tensor Expression，即图中的黄色部分。接下来用户可以手动指定计算策略即scheduler，然后TVM会自动生成特定后端的代码，注意图中的tiling和binding分别代表拆分和绑定的意思，也是scheduler。我们现在明确了scheduler在TVM软件栈中的位置，也应该清楚TVM能否产生高性能的代码关键就在于scheduler是否指定合理，即优化算法在指定后端是否work and efiicient。



![TVM代码生成过程，图源OpenMMLab](https://img-blog.csdnimg.cn/20210327110934662.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)







## 后端优化

前端优化：针对计算图整体拓扑结构优化，不关心算子的具体实现。主要优化流程为对算子节点进行融合、消除、化简，使得计算图的计算和存储开销最小。

[![前端优化示例](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/01Introduction02.png)](https://github.com/chenzomi12/AISystem/blob/main/03Compiler/04Backend/images/01Introduction02.png)

主要是对算子的节点 进行消除融合化简，使得计算图整个的计算，还有存储的开销做到最小。



后端优化：针对单个算子的内部具体实现优化，使得算子的性能达到最优。主要优化流程为对算子节点的输入、输出、内存循环方式和计算逻辑进行编排与转换。

[![后端优化示例](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/01Introduction03.png)](https://github.com/chenzomi12/AISystem/blob/main/03Compiler/04Backend/images/01Introduction03.png)

二者的区别在于关注点不同，前端优化具有局部或全局的视野，而后端优化只关注单个算子节点。



后端部分



![image-20240611150845966](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611150845966.png)

后端优化的流程一般分为三步：

1. 生成低级 IR：将高级或计算图级别 IR（Graph IR）转换为低级 IR（Tensor IR）。

不同 AI 编译器内部低级 IR 形式和定义不同，但是对于同一算子，算法的原理实质相同。对于每个具体的算子，需要用 AI 编译器底层的接口来定义算法，再由编译器来生成内部的低级 IR。

[![生成低级 IR](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/01Introduction04.png)](https://github.com/chenzomi12/AISystem/blob/main/03Compiler/04Backend/images/01Introduction04.png)

2. 后端优化：进行后端优化，并将 IR 转换为更低级的 IR。

针对不同的硬件架构/微架构，不同的算法实现的方式有不同的性能，目的是找到算子的最优实现方式，达到最优性能。同一算子不同形态如 Conv1x1、 Conv3x3、 Conv7x7 都会有不同的循环优化方法。实现方式多种多样，可以人工凭借经验手写算子实现，也可以通过自动调优搜索一个高性能实现。传统编译器如 GCC、LLVM 也具有后端优化的部分，为什么不直接将 AI 编译器的后端优化委托给传统编译器来实现呢？

有两个关键原因：1）数据形式不同：深度学习中数据形式主要为张量（Tensor）。而传统编译器不擅长对张量计算优化，更擅长对标量进行计算。2）缺乏必要的支持：传统编译器主要针对通用编程语言，缺乏对领域特定语言 DSL 的支持，特别是对神经网络，以及相关的特殊优化。

3. 代码生成：根据硬件进行代码生成。

对优化后的低级 IR 转化为机器指令执行，现阶段最广泛的做法为借助成熟的编译工具来实现，代码生成不是 AI 编译器的核心内容。如把低级 IR 转化成为 LLVM、NVCC 等编译工具的输入形式，然后调用其生成机器指令。



> 为什么后端优化不直接用传统的通用编译器，如GCC、LLVM呢?
> 1.深度学习中主要数据为张量(Tensor)传统编译器不擅长对张量计算优化。
> 2.通用编译器主要针对通用编程语言，缺少领域特定语言 DSL支持（特别是神经网络），以及相关的特殊优化。



## 算子优化



### 算子优化的挑战



算子根据其计算形式的特点可分为访存密集型与计算密集型。

1. 访存密集（Memory-Bound）型

指的是在执行过程中主要涉及大量内存读取和写入操作的计算任务。这类算子通常需要频繁地从内存中读取数据，执行一些简单的计算操作，然后将结果写回内存。访存密集型算子的性能受限于内存带宽和访问延迟，而不太受计算能力的限制。如 RNN 训练任务，其网络结构的计算密度很低，因此瓶颈转移到 host 端的 Op Launch 上，算子的计算 kernel 之间出现大量空白。

![image-20240611151731605](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611151731605.png)

2. 计算密集（Compute-Bound）型

指的是在执行过程中主要涉及大量的计算操作，而对内存的访问相对较少的计算任务。这类算子主要依赖于 CPU 或 GPU 的计算能力，并且往往对内存带宽和访问延迟的需求不是特别高。一些数值计算密集型的算法，比如矩阵乘法、卷积运算、复杂的数学函数计算等，通常被认为是计算密集型的操作。

![image-20240611151756741](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611151756741.png)

由于算子种类的多样性，并没有一个一网打尽的优化手段能解决所有算子的高性能执行方式。算子优化存在以下挑战：

- 优化手段多样：要在不同情况下权衡优化及其对应参数，例如针对不同种类算子、相同算子有不同的参数可采用不同优化，对于优化专家来说也是相当耗费精力。
- 通用性与移植性：不同类型的硬件架构差异，使得优化方法要考虑的因素也有很大。例如硬件可使用的指令集，硬件的内存带宽，算力以及存储层次的影响。
- 不同优化间相互影响：各种优化之间可能会相互制约，相互影响。这意味着找到最优的优化方法组合与序列就是一个困难的组合优化问题，甚至是 NP 问题。

算子的不同实现其性能差距千差万别，最好的与最差的相比甚至能达到上百倍的时间开销。为了实现高性能算子，业界有多种做法。

1. 算子库

业界一个最为常见的方式是将预置的算子实现封装成**计算库**。算子库是指一组高度优化的计算核心函数，用于加速特定类型的计算任务，例如常见的矩阵乘法、卷积、循环神经网络等。这些算子库通常是由硬件厂商或第三方开发者编写的，旨在充分利用硬件平台的计算能力，并提供易于使用和高效的接口。

以 CuDNN 为例，它是一个由英伟达公司开发的深度学习加速库，专门针对各种常见的深度学习算法进行了高度优化，使其在英伟达 GPU 上运行时达到最佳性能。CuDNN 中的算子函数使用 CUDA 架构实现，并且在计算时利用了 GPU 硬件的并行性和向量化特性。此外，CuDNN 还通过使用半精度浮点数运算、算法重排等技术来进一步加速计算速度。

类似地，Eigen 是一个由 C++ 编写的线性代数库，用于实现各种矩阵操作，包括矩阵乘法、矩阵求解、特征值分解等。Eigen 中的算子函数使用 SIMD（单指令多数据）指令集实现，并且可以在不同的 CPU 架构上进行自动优化，以提供最佳性能。

这种方法存在三个问题：

- 如何应对 AI 领域算子迭代更新快：AI 领域的算法和模型经常迭代更新，导致算子库需要及时跟进以支持新的算法或模型结构。这可能需要算子库开发者不断更新和优化现有的算子实现，以适应新的需求。
- 如何解决同一算子在多平台移植后一致性问题：算子库通常是为特定硬件平台（如 GPU、CPU）进行优化设计的。但是，在将算子库移植到不同的平台上时，可能会遇到一致性问题。不同平台上的硬件架构和指令集可能存在差异，可能需要进行特定的优化和调整，以确保在多平台上实现一致的计算结果。
- 如何面对算子组合爆炸问题？如参数多样，融合大算子等：在 AI 计算中，经常会遇到大量算子的组合，例如复杂的模型结构或多阶段数据处理流程。这可能导致算子的组合爆炸问题，其中算子之间的参数和组合方式变得多样化和复杂化。

2. 自动生成

那么如何能解决这些问题？是否可以通过自动化生成高性能 kernel 生成的方式来减小算子开发的开销？

目前有两种主流的自动生成算法：

- Auto Tuning：Auto Tuning 是一种通过自动搜索和优化参数组合来生成高效的 kernel 代码的方法。该方法通常基于启发式算法或机器学习技术，自动探索不同参数组合以找到最佳的性能配置。Auto Tuning 可以根据具体的硬件平台和任务特性，自动选择适当的优化策略，从而提高计算核心的性能和效率。
- Polyhedral：Polyhedral 方法是一种基于数学多面体理论的编译优化方法，用于描述循环嵌套的迭代空间和数据依赖关系，并生成高效的循环 kernel 代码。通过对循环迭代空间进行变换和重组，Polyhedral 方法可以实现循环并行化、内存局部性优化等优化，从而提高计算核心的性能和效率。



论文推荐：

![image-20240611153007458](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611153007458.png)

### 算子的计算与调度

- 计算：描述实现算法的具体逻辑，而不关心具体的代码实现。
- 调度：对计算进行优化和控制的过程。通过调度，可以指定计算的执行顺序、内存布局、并行策略等以实现对计算性能的优化。

在神经网络中，深度学习算法由一个个计算单元组成，我们称这些计算单元为算子（Operator，简称 Op）。算子是一个函数空间到函数空间上的映射 ：𝑂：𝑋→𝑌；从广义上讲，对任何函数进行某一项操作都可以认为是一个算子。于 AI 框架而言，所开发的算子是网络模型中涉及到的计算函数。

在神经网络中矩阵乘法是最常见的算子，矩阵乘法的公式为：

$$
C_{ij}=\sum_{k=1}^nA_{ik}\cdot B_{kj}
$$
，其最朴实的实现如下代码：

```
void matrixMultiplication(int A[][128], int B[][128], int result[][128], int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < size; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
```



使用循环分块对其进行优化：循环分块是将大的循环分解为一系列小的循环，以减少内存访问的冲突和提高内存访问的局部性。在矩阵乘法中，我们可以将大的矩阵分解为一系列小的子矩阵，并分别对每个子矩阵进行乘法运算。

```
void matrixMultiplicationTiled(int A[][128], int B[][128], int result[][128], int size, int tileSize) {
    for (int i = 0; i < size; i += tileSize) {
        for (int j = 0; j < size; j += tileSize) {
            for (int k = 0; k < size; k += tileSize) {
                for (int ii = i; ii < i + tileSize; ++ii) {
                    for (int jj = j; jj < j + tileSize; ++jj) {
                        int sum = 0;
                        for (int kk = k; kk < k + tileSize; ++kk) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
                        result[ii][jj] += sum;
                    }
                }
            }
        }
    }
}
```



抑或是使用向量化对其优化：

```
#include <immintrin.h>

void matrixMultiplicationVectorized(int A[][128], int B[][128], int result[][128], int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; j += 4) {
            __m128i row = _mm_set1_epi32(A[i][j]);
            for (int k = 0; k < size; ++k) {
                __m128i b = _mm_loadu_si128((__m128i*)&B[k][j]);
                __m128i product = _mm_mullo_epi32(row, b);
                __m128i currentResult = _mm_loadu_si128((__m128i*)&result[i][j]);
                __m128i updatedResult = _mm_add_epi32(currentResult, product);
                _mm_storeu_si128((__m128i*)&result[i][j], updatedResult);
            }
        }
    }
}
```



我们还可以使用更多的优化方式来实现矩阵乘法，或是将它们组合起来。上面三种操作的算法功能是一样的，但是速度是有差异的。这种差异是和硬件设计强相关的，计算机为加快运算做了许多特殊设计，如存储层次、向量加速器、多个核心等，当我们充分这些硬件特性，可以极大地提升程序执行的速度，优化后的运行效率是原始程序效率的几十倍甚至几百倍。

算子调度具体执行的所有可能的调度方式称为调度空间。AI 编译器优化的目的在于通过对算子进行最佳调度，使得算子在特定硬件上的运行时间达到最优水平。这种优化涉及到对算子调度空间的全面搜索和分析，以确定最适合当前硬件架构的最佳调度方案。这样的优化过程旨在最大程度地利用硬件资源，提高算子的执行效率，并最终实现整体计算任务的高性能执行。





在构建一个算子的调度空间时，首先要确定我们能使用哪些优化手段。同样以 Halide 为例，可以使用的优化有 Reorder(交换)、Split(拆分)、Fuse(融合)、Tile(平铺)、Vector(向量化)、展开(Unrolling)、并行(Parallelizing)等，以 Halide 思想为指导的 AI 编译器 TVM 继承了这些优化方式：

- **Reorder（交换）**：重新排列计算的顺序，可以改变计算的依赖关系，有助于提高缓存命中率，降低内存访问延迟，从而提高性能。
- **Split（拆分）**：将一个维度的计算拆分成多个较小的维度，可以帮助并行化和向量化，并优化内存访问模式。
- **Fuse（融合）**：合并多个计算，减少内存访问和数据传输的开销，提高计算的局部性，以及减少调度开销。
- **Tile（平铺）**：将计算划分成小的块，有利于并行化和向量化，并且可以提高缓存的命中率。
- **Vector（向量化）**：通过将多个数据元素打包成矢量操作，充分利用 SIMD 指令集，提高计算密集型任务的效率。
- **展开（Unrolling）**：循环展开，减少循环的开销，减少分支预测失败的可能性，提高指令级并行性。
- **并行（Parallelizing）**：将计算任务分解成多个并行执行的子任务，充分利用多核处理器或者并行处理单元，提高整体计算速度。

#### 调度树基本概念

对于神经网络中的算子来说，其计算形式一般比较规则，是多层嵌套的循环，也很少有复杂的控制流，并且输入主要是多维张量。分析完计算的特点后，我们来分析下调度的要素。对于一个计算，其首先要进行存储的分配以容纳输入，之后在多层循环下进行计算，得出最终结果后再存储回结果位置。

根据调度的要素，可以将其抽象为一个树结构，称为调度树：

- 循环节点：表示函数如何沿着给定维度进行遍历计算。循环节点与一个函数和一个变量（维度）相关联。循环节点还包含循环是按顺序运行、并行运行还是矢量化运行等信息。
- 存储节点：表示存储待使用的中间结果。
- 计算节点：调度树的叶子，表示正在执行的计算。计算节点可以有其他计算节点作为子节点，以表示内联函数而不是从中间存储加载。

调度树需要满足几个约束才能使调度合法：

- 函数必须在使用之前进行计算：在调度树的深度优先遍历中，函数的计算节点必须出现在其调用函数的计算节点之前。
- 存储必须已分配并在要使用的作用域内：函数的存储节点必须是其计算节点及其调用者的计算节点的祖先。
- 实际代码生成的限制使得某些模式非法。特别是，我们只允许最内层循环（不是任何其他循环节点的祖先的循环节点）的矢量化，并且只允许确定宽度循环的矢量化和展开。

对于任意的算子，可以定义其默认调度。其以行主序的形式遍历所有输出，并且内联所有函数调用，如下图所示：

[![默认调度](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/02OPScompute01.png)](https://github.com/chenzomi12/AISystem/blob/main/03Compiler/04Backend/images/02OPScompute01.png)

我们将调度树与原有的程序进行对应：

[![调度树与程序](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/02OPScompute02.png)](https://github.com/chenzomi12/AISystem/blob/main/03Compiler/04Backend/images/02OPScompute02.png)

在给定一个调度树后，可以通过深度优先搜索的方式进行遍历，然后转换成对应的程序代码：

在给定一个调度树后，可以通过深度优先搜索的方式进行遍历，然后转换成对应的程序代码：

- 如果它是一个循环节点，它将使用适当的属性（并行、矢量化、展开）开始相应的循环，按顺序递归访问每个子节点，并结束相应的循环。
- 如果是存储节点，则分配相应的存储，递归访问每个子节点，并释放存储。
- 如果它是计算节点，则它在其循环定义的域中的位置处计算相应函数的值，并将该值存储在其分配的存储中。

这里就体现计算与调度分离的好处，对于一个计算，可以有多个调度树生成不同性能的程序，只要调度树是合法的，就可以在结果正确的前提下提升程序的性能。

目前主流的方法如 TVM 中采用的是自动调优法。即根据可利用的优化手段，将它们组合，生成一个十分庞大的调度空间，然后利用一些探索器如启发式算法或者机器学习算法，对这个调度空间进行遍历，去实际运行或者用模型预测其性能，根据实际性能反馈对调度空间的探索，最终在一定时间内选择一个相对最优的调度。





### 算子优化

![image-20240611164120802](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611164120802.png)

#### 循环优化

##### 循环展开（有点 跳步 的感觉）

对循环进行展开，以便每次迭代多次使用加载的值，使得一个时钟周期的流水线上尽可能满负荷计算。在流水线中，会因为指令顺序安排不合理而导致NPU等待空转，影响流水线效率。循环展开为编译器进行指令调度带来了更大的空间。



```
循环展开（Loop Unrolling）将一个循环中的多次迭代展开成多个单独的迭代，以减少程序执行的开销，提高代码的运行效率。在计算机执行程序的流水线中，每次跳转到循环体内部都需要进行额外的指令处理和跳转操作，这会增加程序的开销。而循环展开可以通过减少跳转次数、减少指令处理次数等方式，降低指令分支预测的开销来优化程序性能，提高程序执行的速度。

通常循环展开包含以下几个步骤：

1. 复制循环体 n 次，使展开后循环体包括原来循环体 n 个拷贝。这里的 n 一般被称为展开因子

2. 调整数组索引变量的增量以及内存读写操作的地址

3. 删除除了最后一个循环体外的所有循环体中的循环计数变量的增加操作，并且修改最后一个循环体中循环计数变量的增量为原来的 n 倍

4. 删除除了最后一个循环体外的所有循环中的循环条件判断语句

例如原始循环：

```python
for i in range(m):
    a[i] += b[i]
```

通过循环展开，可以将其转换为以下形式：

```
for i in range(0, m-3, 4):
    a[i] += b[i]
    a[i+1] += b[i+1]
    a[i+2] += b[i+2]
    a[i+3] += b[i+3]
for i in range(m-3, m):
    a[i] += b[i]
```



在展开后的循环中，原本执行了 *n* 次循环迭代，变成了执行 *n/4* 次循环展开。

从循环展开的优化有效上分析，循环展开不仅可以减少循环开销，如循环变量测试及分支语句等，还提高了指令之间的并发度，并且因为减少了分支语句从而减少流水线停顿，提升了流水线效率。另外一个分析角度是循环展开后可能会为其他优化提供更多机会。循环展开也有可能会带来负面效果。如果展开后循环体超过指令缓存容量，会引起缓存失效，造成程序性能的下降。并且循环展开会增加寄存器压力，可能导致生成更多的寄存器溢出处理操作，从而降低优化效果。

循环展开最关键的是确定展开因子，目前主要有三种方法：

1. 启发式方法：对循环体代码进行分析，然后使用静态模型计算展开因子。在分析时需要考虑循环本身减少的循环开销、循环展开与其他优化的交互关系等，建立模型需要充分考虑指令级并行度、流水线效率、数据局部性、指令缓存与寄存器的限制等。
2. 根据循环的特征将循环分类，通过大量样本学习，使用分类器建立循环类型和展开因子之间的映射，在实际优化循环时根据循环类型确定最优展开因子。
3. 迭代编译：使用不同展开因子编译生成多个版本的程序并实际运行，选取运行时间最短的作为最优展开因子。

比较上面三个方法可以看到：启发式方法开销最小，展开因子的选择依赖于静态模型的准确性；机器学习开销次之，展开因子的选择不仅依赖于提取的循环特征，而且还需要大量样本进行训练；迭代编译开销最大，但在不考虑开销的情况下肯定可以找到最优展开因子。





**循环展开的优点：**

第一，减少了分支预测失败的可能性。

第二，增加了循环体内语句并发执行的可能性，当然，这需要循环体内各语句不存在数据相关性。

**循环展开的缺点：**

第一，造成代码膨胀，导致 ELF 文件（或 Windows PE 文件）尺寸增大。

第二，代码可读性显著降低，前一个人写的循环展开代码，很可能被不熟悉的后续维护人员改回去。





##### 循环分块

由于内存空间有限，代码访问的数据量过大时，无法一次性将所需要的数据加载到设备内存循环分块能有效提高NPU cache 上的访存效率，改善数据局部性。
如果分块应用于外部循环，会增加计算的空间和时间局部性;分块应与缓存块一起作用，可以提高流水线的效率。

Loop Tiling 的目的是确保一个 Cache 在被用过以后,后面再用的时候其仍然在 cache 中。实现思路:当一个数组总的数据量无法放在 cache 时,把总数据分成一个个 tile 去访问，令每个 tile 都可以满足 Cache
具体做法:把一层内层循环分成 outer loop*inner loop。然后把 outer loop 移到更外层去从而确保 inner loop 一定能满足 Cache

循环分块是利用 Cache 的数据局部性进行优化的一种方法。现代 CPU 通常具有多级 Cache，在存储体系中，Cache 是除 CPU 寄存器外最接近 CPU 的存储层次，相比主存速度更快，但是容量更小。Cache 中复制有 CPU 频繁使用的数据以进行快速访问。由于 Cache 的容量有限，数据会在 Cache 中进行换入换出。当访问的数据在 Cache 中没有时，产生 Cache miss，会向低一级存储层次发出访问请求，然后该数据存储进 Cache，这时访问数据的时间就大大提高。当访问数据就在 Cache 中时，会直接使用该数据以进行复用。

循环分块主要针对大型数据集进行优化，大数据集无法一次全部存入 Cache 中。当遍历该数据集时，循环按照顺序进行访问，会替换掉之前加载进 Cache 的数据，导致后面的指令对之前的数据无法复用，要重新加载数据，产生大量的 Cache miss，数据的复用性很差。程序执行时间变长，大量时间花费在载入数据上。

循环分块将大数据集分成多个小块以充分进行数据复用。数据块的内存访问是一个具有高内存局部性的小邻域。该数据块可以一次加载进 Cache，执行完所有或者尽可能多的计算任务后才被替换出。

在实现中将一层内层循环分成 outer loop * inner loop。然后把 outer loop 移到更外层去，从而确保 inner loop 一定能满足 Cache。

原始的数据存储访问模式和分块后的存储访问模式见下图：

[![循环分块访存模式](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/04LoopOpt02.png)](https://github.com/chenzomi12/AISystem/blob/main/03Compiler/04Backend/images/04LoopOpt02.png)

> 一般处理 CPU/GPU/NPU 都有多级缓存，Tiling如何对应到多级缓存 ?
> AI编译器主要是处理张量，张量的数据排布本来就复杂，人工优化到多级缓存难度高不高?

##### 循环重排

循环重排序（reorder）是矩阵乘法常见的优化方式，指的是对程序中的循环结构重新排列顺序，以优化数据访问模式，特别是在 CNN 中卷积层的应用。通过改变循环的嵌套顺序或者循环内部的迭代顺序，可以改善数据的局部性，减少缓存失效。如下图循环重排序示意图，在矩阵乘法计算中，B 是逐列访问的，在行优先的存储模式下访问模式很不友好。切换内层的循环顺序可以使得所有元素按顺序读取和写入。一次计算输出的一行，得到的是中间结果，全部累加即可得到结果矩阵的一行最终结果，这种方式利用的是内存的空间局部性。

![循环重排访存模式](https://chenzomi12.github.io/_images/04LoopOpt03.png)



内外层循环重排，改善空间局部性，并最大限度地利用引入缓存的数据。对循环进行重新排序以最大程度地减少跨步并将访问模式与内存中的数据存储模式对齐。

![image-20240611200155331](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611200155331.png)











##### 循环融合

![image-20240611200428929](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611200428929.png)

##### 循环拆分

![image-20240611200705152](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611200705152.png)



拆分主要是将循环分成多个循环，可以在有条件的循环中使用，分为无条件循环和含条件循环。

有些处理并行的硬件中，不善于处理控制流。

#### 指令优化

##### 向量化

把矩阵的一些操作当成向量给硬件计算

##### 张量化

![image-20240611204313244](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611204313244.png)

#### 存储优化

##### 访存延迟

![image-20240611204924349](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611204924349.png)

CPU
延迟隐藏可以通过多线程，或者硬件隐式数据预取实现
GPU :
依赖于 Wrap Schedule 对多线程的管理调度和上下文切换实现
NPU/TPU :
采用解耦访问/执行(Decoupled Access/Execute，DAE)架构

在纯串行执行的架构中，计算资源的利用受到显著限制，因为在同一时间点，通常只有一个运算单元处于活跃状态。以程序执行过程为例，其工作流程可概括为以下几个步骤：

1. **数据加载**：首先，系统从主存储器中检索所需数据，并将其加载到处理器的片上缓冲区（on-chip buffer）中。
2. **计算执行**：一旦数据加载完成，计算单元便开始执行预定的计算任务。
3. **数据写回**：最后，这些计算结果被从片上缓冲区写回到主存储器中，以供后续使用或存储。

这种串行处理模式虽然简单，却无法充分利用现代处理器的并行处理能力，导致计算效率和系统性能受限。

在这种情况下，每个运算部件的时间开销都累加得到最终的时延。为了克服这一局限，现代深度学习系统广泛采用并行计算架构，允许多个运算单元同时工作，显著提高了数据处理速度和整体系统性能。延迟隐藏（Latency Hiding）技术在这一领域得到了广泛的应用。该技术通过将内存操作与计算任务并行化，实现了两者的重叠执行，从而最大化了内存带宽和计算资源的利用效率。通过这种方式，即使在数据加载和写回阶段，也能持续执行计算任务，有效减少了因等待内存操作而产生的空闲时间。

CPU 实现延迟隐藏的过程主要依赖于多线程技术和硬件隐式数据预取机制。在多线程环境中，当一个线程等待内存访问时，CPU 可以切换到另一个线程继续执行计算任务，从而减少 CPU 的空闲时间。此外，现代 CPU 通常具备数据预取单元，能够预测程序接下来可能需要的数据，并提前从内存中加载到缓存中，这样当计算单元需要这些数据时，它们已经准备好了，减少了 CPU 等待内存访问的时间。

GPU 在实现延迟隐藏方面，主要依赖于其高度并行化的架构和先进的调度技术。Wrap Schedule 是一种用于管理多线程执行的技术，它允许 GPU 在等待内存操作完成时，动态地调度其他线程来继续执行计算任务。这种技术通过减少线程间的同步开销，提高了线程的执行效率。GPU 还采用了上下文切换机制，能够在不同的线程上下文之间快速切换，进一步隐藏内存访问的延迟。当一个线程因为内存访问而暂停时，GPU 可以立即切换到另一个准备好的线程继续执行，从而保持了 GPU 核心的持续工作。









NPU 采用了解耦访问/执行（Decoupled Access/Execute，DAE）架构。在 DAE 架构中，内存访问操作和计算操作是分开进行的，允许它们并行执行而不是顺序依赖。NPU 拥有专门的硬件单元来处理数据的加载和存储，这些单元独立于执行计算的核心。当计算核心需要数据时，它会发送请求，然后继续执行其他计算任务，而数据加载操作在后台进行。在这种情况下，一般需要使用双缓冲机制，来缓存不同 LOAD 指令得到的数据。

在这种模式下，执行指令会变为并行方式：

![image-20240611205740137](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611205740137.png)



![img](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/05OtherOpt03.png)

##### 存储分配

从传统编译器的视角来看，内存被划分为几个关键区域，每个区域都有其特定的用途和生命周期。

- 局部变量是在程序的函数或代码块内部定义的。它们的存在周期仅限于定义它们的代码块。编译器在调用函数时，会在内存的栈空间中为这些局部变量分配一段内存。当函数执行完毕，这些局部变量的生命周期也随之结束，它们所占用的内存会被自动释放。
- 全局变量在程序的整个生命周期内都是可见的。它们在内存中的静态存储区分配空间，这意味着它们的内存分配在程序启动时完成，并在整个程序运行期间保持不变。全局变量为程序提供了跨函数和代码块的数据共享能力。
- 堆变量是在程序运行时，通过显式请求内存分配而创建的。它们在堆上申请一段内存空间，这为程序提供了极大的灵活性，允许在运行时根据需要动态地增长和缩减内存使用。

在 AI 系统中，这种视角下的内存管理显然无法支撑起 AI 应用。AI 系统通常需要处理大量的数据和复杂的算法，这就需要高效的内存分配和回收策略来支持它们的运行。专用硬件如 GPU 和 NPU 具有各自独特的内存管理机制，这些机制针对它们处理任务的特点进行了优化。

GPU 的内存管理机制包括：

- 全局内存：GPU 拥有自己的内存，称为全局内存或显存，它与 CPU 的内存分开。全局内存是 GPU 中最大的内存区域，但访问延迟较高。
- 共享内存：每个 GPU 线程块（block）可以访问的快速内存，通常比全局内存小得多，但访问速度更快。
- 寄存器：每个线程可以访问的非常快速的内存，但容量有限。
- 常量内存和纹理内存：这些是特殊类型的内存，它们可以提供对数据的缓存访问，以提高性能。

NPU 的内存管理机制包括：

- 片上内存：NPU 通常具有片上内存，用于存储权重和激活等数据，以减少与外部内存的通信开销。
- 内存访问模式：NPU 针对 AI 工作负载进行了优化，支持高并发的内存访问模式。
- 量化和压缩：使用数据量化和压缩技术，可以减少内存占用并提高能效。
- 专用内存控制器：NPU 可能具有专用的内存控制器，用于优化数据流和减少延迟。









#### AutoTuning

对于给定的程序和目标架构，找到最优的编译优化方法。使用哪些优化方法?选择什么参数集?
用什么顺序应用优化方法以达到最佳性能?

优化选择 Optimize Selection
优化顺序 Optimize Sequence

更低实验开销:1)聚焦算子或者 kernel 级别的优化，而非整个程序。2)Cost Model 在CPU上模拟NPU执行，训练和推理推理的模拟速度要求足够快。
特定领域结构:针对神经网络算子或者 kernel 级别的优化，|)主要是高度循环化、张量化、并行化特点进行优化;2)大量相类似的算子计算模式。



##### Ansor

AutoTVM 需要事先编写模板来组成调度的搜索空间，最佳性能的上限取决于模板的设计，这对模板的编写带来了很高的要求。作为第二代调优系统，Ansor（Auto Scheduler）取消了模板机制，优化过程可以自动、无干预的进行：无需手动指定优化方式，编译器自动应用调度原语。

Ansor 自动生成一个覆盖全面的优化的大搜索空间，并为空间中的每个张量程序提供被选择的机会。首先，它自动构建一个大的搜索空间，以覆盖给定计算定义的尽可能多的张量程序。其次，在大搜索空间中高效搜索，该搜索空间可能比现有模板可以覆盖的范围大几个数量级。最后，在优化具有许多子图的整个 DNN 时，识别对端到端性能至关重要的子图并对其进行优先级排序，因为资源是有限的，应该将调优时间和算力资源分配给对性能有更大影响的子图。

![image-20240611215520076](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20240611215520076.png)

Ansor 有三个关键设计，分别是程序采样器、性能微调器、任务调度器。

###### 程序采样器



为了在无模板的前提下自动生成搜索空间，递归地应用一组推导规则来扩展搜索空间；为了避免在搜索空间中陷入局部最优，使用随机抽取完整的程序给每个采样点相同概率。将搜索空间定为两级，高层结构称为草图（sketch），低级结构（分块大小、并行等）称为注解（annotation）。

递归地应用一组派生规则来生成草图，例如在 CPU 上使用这样一组规则来进行草图生成：

- IsStrictInlinable(S,i)：表示 S 中的节点是否是个简单的逐元素算子如 relu，则可以内联
- HasDataReuse：表示节点 i 是否是计算密集型，并且有丰富的算子内部数据重用机会如 matmul、conv2d
- HasFusibleConsumer：表示 S 中的节点 i 是否只有一个节点 j，节点 j 可以融合到节点 i（如 matmul+bias_add，conv2d+relu
- HasMoreReducetionParallel：表示节点在空间维几乎没有并行性，但是在 reduce 维有足够的并行性。（如计算一个矩阵 l2 范数，matmul 2x512 . 512x2）

对计算的定义进行静态分析，获得这些条件谓词的值。这个过程是解析计算的数学表达式的读写模式自动完成的。与 AutoTVM 中手写模板不同，手写模板同时指定了高层规则和低层规则，而草图只有高层结构。

草图只有分块结构，没有分块大小和循环标注如并行、展开和向量化，这部分由标注完成。给定一个草图列表，随机在草图上填充分块大小、对循环进行随机标注。

###### 性能微调器

使用进化搜索和学习成本模型来微调采样程序的性能。

- 进化搜索

在高质量程序的基础上进行突变。突变类型包括分块大小、并行、计算位置等。

- 成本模型

基于梯度增强决策树作为底层模型

###### 任务调度器

一个 DNN 可以被划分为许多独立的子图，对于某些子图，花费时间对他们进行调优并不能显著提升整个 DNN 的端到端的性能。为了提高调优效率，Ansor 动态的将不同的时间资源进行分配。

以优化单个 DNN 的端到端延迟为例，Ansor 对具有较高初始延迟的子图进行优先排序，因为乐观猜测可以快速减少其延迟。之后，如果 Ansor 花费了多次迭代而没有观察到其延迟的减少，那么 Ansor 就会离开子图。

Ansor 也存在诸多限制，例如不能对动态形状的图进行优化，也无法使用硬件平台特定指令支持，如 Intel VNNI、NVIDIA Tensor Core。

##### Meta Scheduler

Meta Schedule 是第三代调优系统，在它之前，TVM 存在三套生成调度的 API：

- 手动调度：开发人员利用调度原语手动优化程序，程序性能高度依赖开发者的经验。
- AutoTVM：开发者需要为每个算子设计一个调度生成的模板，从而使得调优器可以探索这个生成的调度空间。
- Ansor：根据一组预定义的规则，自动生成调度模板作为设计空间。但是扩展 Ansor 到新的调度原语如张量化、流水线化等绝非易事。

上面三个系统都有独立的 API，且互不兼容。AutoTVM 要求用户学习一组新的 API，AutoScheduler 引入了另一组基于 C++ 的搜索规则。它增加了用户的心理负担和扩展现有系统的开销。

MetaSchedule 提供以下特性：

- 用于实现手动调优、AutoTVM 风格和 AutoScheduler 风格的统一 API。
- 所有调度原语的可扩展性，包括张量化和循环分块。在自动调优中使用新的原语几乎不需要额外的努力。
- 自动化基础设施在其每个组件上都是可扩展的。每个组件的系统可以在纯 python 或 C++或两者中轻松自定义。例如，可以开发一个新的在 python 中的调度空间生成器或者新的 ProgramRunner 等。

Meta Scheduler 遵循下图的调优流程：

[![img](https://github.com/chenzomi12/AISystem/raw/main/03Compiler/04Backend/images/06AutoTuning03.png)](https://github.com/chenzomi12/AISystem/blob/main/03Compiler/04Backend/images/06AutoTuning03.png)

###### 设计空间生成



底层系统记录了用户使用的所有调度原语，以及在采样指令上做出的随机决策，称为 Trace。Trace 可能包含零条或多条采样指令，这些指令引入调度中的不确定性，形成了一个可供探索的设计空间，例如哪一组分块大小在特定硬件上效果最佳。

###### 探索设计空间



Meta Schedule 提供了多种内置的探索策略，可以详尽或高效地进行搜索 ，来实现高效的调度。其搜索策略与之前调优器的搜索策略类似，可以随机搜索，也可以使用成本模型进行指导。

###### 数据库

所有度量记录都经过序列化并存储在数据库中。数据库记录的内容包括工作负载，序列化的 TensorIR;执行测量的硬件目标;参数类型：输入张量的形状和 dtype；运行时间等。

在 Meta scheduler 中，成本模型、数据库、特征提取器、程序运行器等都是可定制、易于扩展的。

















# 

# 动态 shape 的挑战与解决现状





概述动态 shape 是什么一般是 2 个问题：Tensor 的 shape 是动态的。Control flow 的动态性。即 if-else / loop 类的结构。比如，where，while，Switch 等。注：PyTorch 的动态图，与动态 shape 是 2 个不同的概念。Tensor shape 的动态性，类型比较多。比如：变化的 batch size。变化的 input shape。CV 模型的 image resolution，NLP 模型的 input sequence length。算子逻辑导致的。unique，nonzero。稀疏矩阵导致的。输入可以是静态的，但为了节约存储空间，使用稀疏矩阵表示，成了动态 shape。搜推模型用的多。动态 shape 与 Control flow 相互影响。其他特殊情况。比如，random 算子、range 算子，可能间接导致动态 shape。解决方案，集中在 compiler 层面。核心是如何 static 化，基本靠 padding。动态 shape 的挑战Compiler 不擅长处理动态 shape。静态 shape 语义下比较确定性的问题，动态 shape 场景很复杂。比如：算子占用的显存预估 & 调度策略，是否需要 implicit broadcast，[https://numpy.org/doc/stable/user/basics.broadcasting.html](http://link.zhihu.com/?target=https%3A//numpy.org/doc/stable/user/basics.broadcasting.html)指令层的向量化，codegen 模版选择。Broadcast 的例子A one dimensional array added to a two dimensional array results in broadcasting![img](https://pic1.zhimg.com/80/v2-7bbf8c3a54a27dd347cc9a4bf1f1d228_1440w.webp?source=d16d100b)
Hardware 层面的困难：产生更多的数据依赖、跳转/条件分支等指令，导致更多的 流水线气泡 (pipeline stall)。codeine 时向量化的指令变少，生成更多的标量计算、稀碎的数据搬移指令。对 GPU / DSA 不友好。（还有别的吗？）pipeline stall on Control Flow 的例子![img](https://picx.zhimg.com/80/v2-aee6e13bdba7e2133825fe92a6e11bf2_1440w.webp?source=d16d100b)解决思路分 3 大类。software 层面，主要是 compiler 相关。核心是如何 static 化，基本靠 padding。hardware 层面，改进芯片设计。比如，control flow 改进、pipeline 改进、ISA 设计等。不解决不优化，用 cpu 慢慢跑。软件的解决方案“解决” 基本是不可能的，只能 case by case 的绕过去。图片来自 [AI编译优化--Dynamic Shape Compiler](https://zhuanlan.zhihu.com/p/305546437)![img](https://picx.zhimg.com/80/v2-1b67974a5a1cba3b8ca2ae2d3a1262cb_1440w.webp?source=d16d100b)1. Padding 成 static shapePadding 也分两种情况：模型的 input，可以在 CPU 上做 padding，灵活性很高，方案很多。Model 内部算子的 input，padding 不好做，有需求，但相对也少。模型 input 的常见 padding 方案：方案说明分析Naive padding把变化的维度，padding 到最大长度。- 问题：Shape length 变化很大时，计算效率非常低。
\- 真实场景下，大部分 input 远小于 max length。多个模型，分桶准备 N 个模型，最大长度 64, 128，512, 2048。选最小可用的模型，padding 到对应长度。- 使用最广泛。一般，模型只需要训练一个最大的即可。
\- 问题：常驻的显存增加 N 倍，上层的调度系统工作量增加。
\- 适用范围，主要针对模型输入的动态 shape。不适内部算子的动态 shape，特别是，动态算子较多时，产生的 shape combination 爆炸。
\- 需要提前调研和评估，从而选定模型尺寸。动态编译第一次执行时，不优化。执行后，根据真实 shape 编译出高性能版本。下次遇到符合的 shape，使用高性能版本。- 一些 AI compiler、算子库开始采用。比如，华为的 cann。
\- 不用提前调研用户需求。
\- 算法要求比较高，不能保存太多的 compile cache。否则，内存占用、query overhead 等，更复杂。crop 数据针对特别长的 input，数据截断不处理。- 作为系统健壮性的兜底方案，一般都会做，区别只是 threshold 的大小。
\- 很多的 crop，基本不影响算法精度。比如，cv 场景把 225*226 的图片 crop 成 224 * 224。Bert 模型按 max length 512 截断。很可能，不影响模型在该条数据上最终输出。
\- 精度受影响的 input，只要占比足够低，业务上都能接受。Padding 的主要难点引入较少/更少的无效数据。如表格里的方案。除了占用 memory / bandwidth，也可能增加 compute 时间。Padding 策略不能（明显）影响模型的精度表现。新策略，一般需要数学、算法实验论证。有工作量和研发成本。尺寸的局限性。以 CV 模型为例，700*700 的图片，padding 成 800*800，一般是可以的，但如果 padding 成 1600*1600，算法精度可能急剧下降。Padding 导致的内存搬运 or 不连续。Tensor 一般是高维的，但内存里通常是一维的排列。比如，灰度图的长宽是 2 维，RGB image 的 3 个 channel。如果 Padding 在 CPU 做，上述困难就简单很多。CPU 的计算通常不是瓶颈，也擅长处理 padding 问题。padding 的变化可能丰富，如下图。以二维数据（image）为例，多种 padding 策略，算法精度、计算效率的影响是很丰富的。![img](https://picx.zhimg.com/80/v2-be8c70e6089e03b364c1d60d024ac343_1440w.webp?source=d16d100b)2. 消除“伪”动态 shape很多看似是动态的模型，在推理时，完全等价于静态模型。推理框架、AI compiler 可以自动的静态化，模型精度无损失。典型场景：动态 batch size 引入 shape 等额外算子。如果固定 batch size，可以消除掉动态性。能静态化的算子。比如，flatten, gather, expand, slice 等。以 flatten 为例，见下表。flatten 有 2 个输入：data 和 axis。axis 取值不一样，Output shape 很可能会变，即，动态 shape。推理时，算子的 axis 一般是 constant 参数，不会变。即，可以处理成静态 shape。Data (Input) shapeaxisOutput shape(2, 3, 4, 5)0(1, 120)1(2, 60)2(6, 20)3(24, 5)3. Shape Constraint - 缩小动态的范围推理框架、AI compiler 尝试的另一个方向。有些“真”动态的算子，可以通过前后算子推演出 shape 的约束关系。AI compile 知道算子内存的 upper bound，可以明显缩小搜索空间。比如：Int8 推理时，unique 算子的输出，shape 的 upper bound 是 min(input length, 255)。A + B = C 场景，Add 算子的输入 vector A 和 vector B，shape 是相同的。4. 改代码、换算子有点难，普适性也不高。但，确实有一些能用的成熟规则。来自 habana 的例子：[https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Dynamic_Shapes.html#mitigation-techniques-for-dynamic-ops](http://link.zhihu.com/?target=https%3A//docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Dynamic_Shapes.html%23mitigation-techniques-for-dynamic-ops)![img](https://pic1.zhimg.com/80/v2-9c7aa8651ac835d87f71bf700903a21f_1440w.webp?source=d16d100b)5. Control Flow 的支持分 2 种情况：尝试消除 Control flow。基本就是编译原理、cpu pipeline 中 control flow 消除的方法。比如，loop unrolling (循环展开), Polyhedral Model （多面体模型）消除不掉的：if-else、switch 等条件选择分支，每个分支都编译，cache 起来。runtime 动态选择。Loop / while 等结束条件动态的循环，暂不知道加速方法。也比较少见。以 Polyhedral Model 为例，思路如下：当 if...else / loop 的判断条件 (cond) 全部已知时，影响判断条件的 N 个变量，所有取值构成一个 N 维空间。实际触发 compute 计算的，是 N 维空间里的一个 subset 子集。Compilation time 可以精确枚举 subset 子集，transform 成 1 层的 loop 循环。然后 loop unrolling (循环展开)，即，消掉了所有的 control flow 指令。详见 [https://www.cse.iitk.ac.in/users/swarnendu/courses/autumn2020-cs610/intro-to-polyhedral-model.pdf](http://link.zhihu.com/?target=https%3A//www.cse.iitk.ac.in/users/swarnendu/courses/autumn2020-cs610/intro-to-polyhedral-model.pdf)![img](https://picx.zhimg.com/80/v2-65cd38b7cfef931dfd17df0b8b3700d2_1440w.webp?source=d16d100b)6. Sequence Mask实现参考 [https://github.com/bytedance/effective_transformer](http://link.zhihu.com/?target=https%3A//github.com/bytedance/effective_transformer)思路：发现：模型的部分计算，与 token 的位置和上下文无关。可以把多个短 input 数据 concat 成 1 个长的，推理结果不变。方案：注册 TransformerInput、Transformer、TransformerOutput 三个算子，分别进行 input 数据标记、静态图计算、output 数据恢复。![img](https://picx.zhimg.com/80/v2-0bc9877ff26349c662ac8a4bbef5d524_1440w.webp?source=d16d100b)相关 AI compiler 调研AI compiler 总结主要看了 3 条线：NV 的 TensorRTMLIR。阿里的 BladeDISC。TVM 方向。relax，Amazon 的 Nimble 和 DietCode 2 个工作。从文章看，阿里的 BladeDISC 系列感觉最好，理解最深刻。没有实测过，真实情况不了解。其他 compiler:HAOTuner: A Hardware Adaptive Operator Auto-Tuner for Dynamic Shape Tensor Compilers [https://ieeexplore.ieee.org/document/10160123](http://link.zhihu.com/?target=https%3A//ieeexplore.ieee.org/document/10160123)[https://github.com/merrymercy/awesome-tensor-compilers](http://link.zhihu.com/?target=https%3A//github.com/merrymercy/awesome-tensor-compilers)具体的实现上，主要是 3 类工作。前 2 个是辅助，最后一个是目的。Shape 的语义编码 & IR 中的传递。shape function。运行时的 shape inference、shape 估算、上下界约束。调度。先把 static shape 的算法，迁移到动态 shape 上。然后再努力能设计点更厉害的。单算子分：计算密集型、访存密集型。调度算法有差异。基本都是拿时间换空间，比如，申请个稍微偏大的内存（冗余计算）、保持多个版本的 cache（知识库）。解决方案，本质都是：静态化。区别只是 coding 的工作量、overhead 的来源和大小。主要是 2 种静态化方法：Padding预分配更大的内存，比如，NMS。数学的说，static shape 是 global observationDynamic shape 是 partial observation，部分可观察。难以实现全局最优解。因此，求解的方法和难度，完全不一样。TensorRT - Padding没有详细调研。根据 [杨军 2022 年的文章](https://zhuanlan.zhihu.com/p/463629676) 描述，TensorRT 至今的动态 shape 方案，依然就是 padding。DISC - 阿里 基于 MLIRDISC：A Dynamic Shape Compiler for Machine Learning Workloads[https://browse.arxiv.org/pdf/2103.05288.pdf](http://link.zhihu.com/?target=https%3A//browse.arxiv.org/pdf/2103.05288.pdf)[https://github.com/alibaba/BladeDISC](http://link.zhihu.com/?target=https%3A//github.com/alibaba/BladeDISC)个人感觉最好的工作。2022 年，在 Transformer 中的加速比为 1.8 倍，优于 Nimble。Nvidia T4 硬件上几个真实的业务例子的性能收益数字：![img](https://picx.zhimg.com/80/v2-af4c7b393b9579a1caca0f88a74a5718_1440w.webp?source=d16d100b)海光 DCU 上几个真实业务例子上的性能数字：某识别类模型推理不同batchsize下 2.21X ～ 2.31X某检测类模型A推理不同batchsize下 1.73X ～ 2.1X某检测类模型B推理不同batchsize下 1.04X ～ 1.59X某分子动力学模型训练2.0XTVM - Relaxpaper 解读: [TVM Relax 如何支持 dynamic shape](https://zhuanlan.zhihu.com/p/627449108)个人看法：主要是解决 TVM 体系中的 shape 语义编码和 shape function。并没有直接解决任何模型的动态 shape 调度问题。距离实际产生价值，还有不少的工作量。Relax 只解决静态 rank 的动态 shape。不能解决 rank 也动态的情况。理解 shape 语义编码的好例子`@R.function def shape_example(x: R.Tensor[(n, 2, 2), "f32"]):    with R.dataflow(): lv0: R.Tensor[(n, 4), "f32"] = R.reshape(x, (n, 4))  lv1: R.Tensor[(n * 4,), "f32"] = R.flatten(lv0) lv2: R.Shape = (n * 4,) lv3: R.Shape = R.call_packed("myshape_func", lv2) `解读：shape 在 IR 中单独作为表达式存在了（lv2），以前在 relay 中只能是 tensor 的附属；shape 中可以包含 “n” 这样的未知变量了，运行时确定。编译时可推断：R.Tensor[(n, 4), "f32"] 的存储空间是 R.Tensor[(n, 2), "f32"] 的 2 倍。Nimble - AWS 基于 TVM亮点是，做了个基于 Virtual Matchine 的轻量级跨平台 runtime，将动态 shape 模型的运行时控制逻辑预构建为 VM 解释执行。TODO - 这个 VM 是什么，怎么帮助改进动态 shape 问题的，没细看，不懂。支持动态 shape 的额外开销：相比 TVM 静态模型性能降低 5%~25%，性能降低主要来自于动态 shape 算子 index 的计算和 VM 引入的指令开销；相比 TVM 额外占用 8% 内存Dense 算子 Full Dispatch 性能基本持平静态 shape，No Dispatch 性能最差。DietCode - AWS 基于 TVMpaper: [https://assets.amazon.science/14/33/43345d8142d8936ec591f5600aa5/dietcode-automatic-optimization-for-dynamic-tensor-programs.pdf](http://link.zhihu.com/?target=https%3A//assets.amazon.science/14/33/43345d8142d8936ec591f5600aa5/dietcode-automatic-optimization-for-dynamic-tensor-programs.pdf)paper 解读: [DietCode：TVM中新的动态Shape解决方案](https://zhuanlan.zhihu.com/p/590531033)TVM-RFC：[tvm-rfcs/0072-dynamic-autoscheduler.md at main · apache/tvm-rfcs](http://link.zhihu.com/?target=https%3A//github.com/apache/tvm-rfcs/blob/main/rfcs/0072-dynamic-autoscheduler.md) 代码应该还没入 TVM [github PR](http://link.zhihu.com/?target=https%3A//github.com/apache/tvm/issues/11516)主要提出了 micro-kernel 的概念来解决动态 Shape 在TVM中自动调度的问题，极大减少了搜索时间。本质上仍然是一种“分组”策略，将动态问题转化为静态问题。一般，“分组”概念停留在模型层面。DietCode 将“分组”下沉，节省大量的存储空间，并自动搜索、降低搜索时间。以 BERT 模型为例，DietCode 与 Ansor 相比， 搜索时间减少 5.88倍，性能提高约 70%
