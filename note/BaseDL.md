# 卷积神经网络 CNN



## 卷积相关计算

输入通道个数 等于 卷积核通道个数

卷积核个数 等于 输出通道个数 

> 多通道卷积过程，应该是输入一张三通道的图片，这时有多个卷积核进行卷积，并且每个卷积核都有三通道，分别对这张输入图片的三通道进行卷积操作。每个卷积核，分别输出三个通道，这三个通道进行求和，得到一个featuremap，有多少个卷积核，就有多少个featuremap





1x1卷积核降维？

有哪些指标？

L1，L2正则化？



## Inception

1. `Inception` 网络是卷积神经网络的一个重要里程碑。在`Inception` 之前，大部分流行的卷积神经网络仅仅是把卷积层堆叠得越来越多，使得网络越来越深。这使得网络越来越复杂，参数越来越多，从而导致网络容易出现过拟合，增加计算量。

   而`Inception` 网络考虑的是**多种卷积核的并行计算**，扩展了网络的宽度。

2. `Inception Net` 核心思想是：稀疏连接。因为生物神经连接是稀疏的。

3. `Inception` 网络的最大特点是大量使用了`Inception` 模块。

   

###  Inception v2

1. `Inception v2` 的主要贡献是提出了`Batch Normalization` 。论文指出，使用了`Batch Normalization` 之后：

   - 可以加速网络的学习。

     相比`Inception v1`，训练速度提升了14倍。因为应用了`BN` 之后，网络可以使用更高的学习率，同时删除了某些层。

   - 网络具有更好的泛化能力。

     在`ImageNet` 分类问题的`top5` 上达到`4.8%`，超过了人类标注 `top5` 的准确率。

2. `Inception V2` 网络训练的技巧有：

   - 使用更高的学习率。
   - 删除`dropout`层、`LRN` 层。
   - 减小`L2` 正则化的系数。
   - 更快的衰减学习率。学习率以指数形式衰减。
   - 更彻底的混洗训练样本，使得一组样本在不同的`epoch` 中处于不同的`mini batch` 中。
   - 减少图片的形变。







## BN和LN

Normalize

为什么要进行Normalize呢？
在神经网络进行训练之前，都需要对于输入数据进行Normalize归一化，目的有二：1，能够加快训练的速度。2.提高训练的稳定性。

为什么使用Layer Normalization（LN）而不使用Batch Normalization（BN）呢？

先看图，LN是在同一个样本中不同神经元之间进行归一化，而BN是在同一个batch中不同样本之间的同一位置的神经元之间进行归一化。
BN是对于相同的维度进行归一化，但是咱们NLP中输入的都是词向量，一个300维的词向量，单独去分析它的每一维是没有意义地，在每一维上进行归一化也是适合地，因此这里选用的是LN。



一、batch normalization
batch normalization是对一批样本的同一纬度特征做归一化。如下图我们想根据这个batch中的三种特征（身高、体重、年龄）数据进行预测性别，首先我们进行归一化处理，如果是Batch normalization操作则是对每一列特征进行归一化，如下图求一列身高的平均值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/34a7f4320aab432091b9c7abc45a8ef4.png#pic_center)

BN特点：强行将数据转为均值为0，方差为1的正态分布，使得数据分布一致，并且避免梯度消失。而梯度变大意味着学习收敛速度快，能够提高训练速度。

设`batch_size`为m，网络在向前传播时，网络 中每个神经元都有m个输出，BN就是将每个神经元的m个输出进行归一化处理，看到BN原论文中的伪代码：
![在这里插入图片描述](https://img-blog.csdnimg.cn/dce7fb3f51fa4d7fb781260fa68d5608.png#pic_center)



![1aceb7c7a386b66b8f04211f2531ce0](F:\wechat_cache\WeChat Files\wxid_yginngtqpyfw22\FileStorage\Temp\1aceb7c7a386b66b8f04211f2531ce0.png)

二、layer normalization
而layer normalization是对单个样本的所有维度特征做归一化。如下表中，如果是Layer normalization则是对每一行（该条数据）的所有特征数据求均值。



![在这里插入图片描述](https://img-blog.csdnimg.cn/ec0df07e39ab45eaaa0139ffe0ac176b.png#pic_center)

三、应用场景
3.1 两者的区别
从操作上看：BN是对同一个batch内的所有数据的同一个特征数据进行操作；而LN是对同一个样本进行操作。
从特征维度上看：BN中，特征维度数=均值or方差的个数；LN中，一个batch中有batch_size个均值和方差。

如在NLP中上图的C、N、H,W含义：
N：N句话，即batchsize；
C：一句话的长度，即seqlen；
H,W：词向量维度embedding dim。



3.2 BN和LN的关系
BN 和 LN 都可以比较好的抑制梯度消失和梯度爆炸的情况。BN不适合RNN、transformer等序列网络，不适合文本长度不定和batchsize较小的情况，适合于CV中的CNN等网络；
而LN适合用于NLP中的RNN、transformer等网络，因为sequence的长度可能是不一致的。
栗子：如果把一批文本组成一个batch，BN就是对每句话的第一个词进行操作，BN针对每个位置进行缩放就不符合NLP的规律了。
3.3 小结
（1）经过BN的归一化再输入激活函数，得到的值大部分会落入非线性函数的线性区，导数远离导数饱和区，避免了梯度消失，这样来加速训练收敛过程。
（2）归一化技术就是让每一层的分布稳定下来，让后面的层能在前面层的基础上“安心学习”。BatchNorm就是通过对batch size这个维度归一化来让分布稳定下来（但是BN没有解决ISC问题）。LayerNorm则是通过对Hidden size这个维度归一。



## ResNet

1. `ResNet` 提出了一种残差学习框架来解决网络退化问题，从而训练更深的网络。这种框架可以结合已有的各种网络结构，充分发挥二者的优势。

2. `ResNet`以三种方式挑战了传统的神经网络架构：

   - `ResNet` 通过引入跳跃连接来绕过残差层，这允许数据直接流向任何后续层。

     这与传统的、顺序的`pipeline` 形成鲜明对比：传统的架构中，网络依次处理低级`feature` 到高级`feature` 。

   - `ResNet` 的层数非常深，高达1202层。而`ALexNet` 这样的架构，网络层数要小两个量级。

   - 通过实验发现，训练好的 `ResNet` 中去掉单个层并不会影响其预测性能。而训练好的`AlexNet` 等网络中，移除层会导致预测性能损失。

3. 很多证据表明：残差学习是通用的，不仅可以应用于视觉问题，也可应用于非视觉问题。

### 残差块

![image-20241027092657309](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20241027092657309.png)

## SENet

1. `SENet` 提出了一种新的架构单元来解决通道之间相互依赖的问题。它通过显式地对通道之间的相互依赖关系建模，自适应的重新校准通道维的特征响应，从而提高了网络的表达能力。
2. `SENet` 以`2.251% top-5` 的错误率获得了`ILSVRC 2017` 分类比赛的冠军。
3. `SENet` 是和`ResNet` 一样，都是一种网络框架。它可以直接与其他网络架构一起融合使用，只需要付出微小的计算成本就可以产生显著的性能提升。

### SE块

![image-20241027095209202](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20241027095209202.png)



#### squeeze 操作

1. `squeeze` 操作的作用是：跨空间 聚合特征来产生通道描述符。

   该描述符嵌入了通道维度特征响应的全局分布，包含了全局感受野的信息。

2. 每个学到的滤波器都是对局部感受野进行操作，因此每个输出单元都无法利用局部感受野之外的上下文信息。

   在网络的低层，其感受野尺寸很小，这个问题更严重。

   ![img](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet_squeeze.png)

   为减轻这个问题，可以将全局空间信息压缩成一组通道描述符，每个通道对应一个通道描述符。然后利用该通道描述符。

   ![img](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet_squeeze2.png)

   ![image-20241027100521541](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20241027100521541.png)

#### 7.1.2 excitation 操作

1. `excitation` 操作的作用是：通过自门机制来学习每个通道的激活值，从而控制每个通道的权重。

2. `excitation` 操作利用了`squeeze` 操作输出的通道描述符 z 。

   - 首先，通道描述符 z 经过线性降维之后，通过一个`ReLU` 激活函数。

     降维通过一个输出单元的数量为 C/r 的全连接层来实现，其中 r 为降维比例。

   - 然后，`ReLU` 激活函数的输出经过线性升维之后，通过一个`sigmoid` 激活函数。

     升维通过一个输出单元的数量为C 的全连接层来实现。

   通过对通道描述符z 进行降维，显式的对通道之间的相互依赖关系建模。

   ![img](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet_excitation.png)

![image-20241027100418605](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20241027100418605.png)


#### 





##  DenseNet



<img src="https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/denset_1.png" alt="img" style="zoom: 50%;" />



1. `DenseNet`不是通过更深或者更宽的结构，而是通过特征重用来提升网络的学习能力。

2. `ResNet` 的思想是：创建从“靠近输入的层” 到 “靠近输出的层” 的直连。而`DenseNet` 做得更为彻底：将所有层以前馈的形式相连，这种网络因此称作`DenseNet` 。

3. `DenseNet` 具有以下的优点：

   - 缓解梯度消失的问题。因为每层都可以直接从损失函数中获取梯度、从原始输入中获取信息，从而易于训练。
   - 密集连接还具有正则化的效应，缓解了小训练集任务的过拟合。
   - 鼓励特征重用。网络将不同层学到的 `feature map` 进行组合。
   - 大幅度减少参数数量。因为每层的卷积核尺寸都比较小，输出通道数较少 (由增长率  决定)。

4. `DenseNet` 具有比传统卷积网络更少的参数，因为它不需要重新学习多余的`feature map` 。

   - 传统的前馈神经网络可以视作在层与层之间传递`状态`的算法，每一层接收前一层的`状态`，然后将新的`状态`传递给下一层。

     这会改变`状态`，但是也传递了需要保留的信息。

   - `ResNet` 通过恒等映射来直接传递需要保留的信息，因此层之间只需要传递`状态的变化` 。

   - `DenseNet` 会将所有层的`状态` 全部保存到`集体知识`中，同时每一层增加很少数量的`feture map` 到网络的`集体知识中`。

5. `DenseNet` 的层很窄（即：`feature map` 的通道数很小），如：每一层的输出只有 12 个通道。

## 内存优化

深度学习的内存怎么理解？

### 内存消耗

1. 虽然 `DenseNet` 的计算效率较高、参数相对较少，但是`DenseNet` 对内存不友好。考虑到`GPU` 显存大小的限制，因此无法训练较深的 `DenseNet` 。

2. 假设`DenseNet`块包含 $L$层，对于第$l$ 层有：$X_l = H_l([x_0, x_1, ...,x_{l-1}])$ 。

   假设每层的输出`feature map` 尺寸均为$W \times H$、通道数为$k$ ， 由`BN-ReLU-Conv(3x3)` 组成，则：

   - 拼接`Concat`操作 $[...]$ ：需要生成临时`feature map` 作为第  $l$层的输入，内存消耗为 $WHk \times l $ 。
   - `BN` 操作：需要生成临时`feature map` 作为`ReLU` 的输入，内存消耗为 $WHk \times l $ 。
   - `ReLU` 操作：可以执行原地修改，因此不需要额外的`feature map` 存放`ReLU` 的输出。
   - `Conv` 操作：需要生成输出`feature map` 作为第 $l$ 层的输出，它是必须的开销。

   因此除了第 $1,2,···，L$层的输出`feature map` 需要内存开销之外，第 层还需要 的内存开销来存放中间生成的临时`feature map` 。

   整个 `DenseNet Block` 需要 $WHk (1+L)L $的内存开销来存放中间生成的临时`feature map` 。即`DenseNet Block` 的内存消耗为$o(L^2)$，是网络深度的平方关系。

3. 拼接`Concat`操作是必须的，因为当卷积的输入存放在连续的内存区域时，卷积操作的计算效率较高。而`DenseNet Block` 中，第 层的输入`feature map` 由前面各层的输出`feature map` 沿通道方向拼接而成。而这些输出`feature map` 并不在连续的内存区域。

   另外，拼接`feature map` 并不是简单的将它们拷贝在一起。由于`feature map` 在`Tensorflow/Pytorch` 等等实现中的表示为$R^{n \times d \times w \times h}$ （`channel first`)，或者$R^{n \times w \times h \times d}$ (`channel last`），如果简单的将它们拷贝在一起则是沿着`mini batch` 维度的拼接，而不是沿着通道方向的拼接。

4. `DenseNet Block` 的这种内存消耗并不是`DenseNet Block` 的结构引起的，而是由深度学习库引起的。因为`Tensorflow/PyTorch` 等库在实现神经网络时，会存放中间生成的临时节点（如`BN` 的输出节点），这是为了在反向传播阶段可以直接获取临时节点的值。

   这是在时间代价和空间代价之间的折中：通过开辟更多的空间来存储临时值，从而在反向传播阶段节省计算。

5. 除了临时`feature map` 的内存消耗之外，网络的参数也会消耗内存。设$H_l$ 由`BN-ReLU-Conv(3x3)` 组成，则第 层的网络参数数量为：$9lk^2$ （不考虑 `BN` ）。

   整个 `DenseNet Block` 的参数数量为 $9k^2(L+1)L/2$ ，即$O(L^2)$ 。因此网络参数的数量也是网络深度的平方关系。

   - 由于`DenseNet` 参数数量与网络的深度呈平方关系，因此`DenseNet` 网络的参数更多、网络容量更大。这也是`DenseNet` 优于其它网络的一个重要因素。
   - 通常情况下都有 $WH > 9k/2$ ，其中 $W,H$ 为网络`feature map` 的宽、高，$k$ 为网络的增长率。所以网络参数消耗的内存要远小于临时`feature map` 消耗的内存。

#### 内存优化

1. 论文`《Memory-Efficient Implementation of DenseNets》`通过分配共享内存来降低内存需求，从而使得训练更深的`DenseNet` 成为可能。

   其思想是利用时间代价和空间代价之间的折中，但是侧重于牺牲时间代价来换取空间代价。其背后支撑的因素是：`Concat`操作和`BN` 操作的计算代价很低，但是空间代价很高。因此这种做法在`DenseNet` 中非常有效。

2. 传统的`DenseNet Block` 实现与内存优化的`DenseNet Block` 对比如下（第 层，该层的输入`feature map` 来自于同一个块中早前的层的输出）：

   - 左图为传统的`DenseNet Block` 的第 $l$层。首先将 `feature map` 拷贝到连续的内存块，拷贝时完成拼接的操作。然后依次执行`BN`、`ReLU`、`Conv` 操作。

     该层的临时`feature map` 需要消耗内存 $2WHkl$，该层的输出`feature map` 需要消耗内存$WHk$ 。

     - 另外某些实现（如`LuaTorch`）还需要为反向传播过程的梯度分配内存，如左图下半部分所示。如：计算 `BN` 层输出的梯度时，需要用到第  层输出层的梯度和`BN` 层的输出。存储这些梯度需要额外的 $O(lk)$ 的内存。
     - 另外一些实现（如`PyTorch,MxNet`）会对梯度使用共享的内存区域来存放这些梯度，因此只需要  $O(k)$的内存。

   - 右图为内存优化的`DenseNet Block` 的第 层。采用两组预分配的共享内存区`Shared memory Storage location` 来存`Concate` 操作和`BN` 操作输出的临时`feature map` 。

   ![img](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_mem_block.png)

3. 第一组预分配的共享内存区：`Concat` 操作共享区。第 $1,2,3...,L$层的 `Concat` 操作的输出都写入到该共享区，第 $l+1$层的写入会覆盖第 $l$层的结果。

   - 对于整个`Dense Block`，这个共享区只需要分配$WHkL$ （最大的`feature map` ）的内存，即内存消耗为 $O(kL)$(对比传统`DenseNet` 的$O(kL^2)$ )。

   - 后续的`BN` 操作直接从这个共享区读取数据。

   - 由于第 $l+1$层的写入会覆盖第 $l$层的结果，因此这里存放的数据是临时的、易丢失的。因此在反向传播阶段还需要重新计算第$l$ 层的`Concat` 操作的结果。

     因为`Concat` 操作的计算效率非常高，因此这种额外的计算代价很低。

4. 第二组预分配的共享内存区：`BN` 操作共享区。第$1,2,3,...,L$ 层的 `BN` 操作的输出都写入到该共享区，第$l+1$ 层的写入会覆盖第 $l$层的结果。

   - 对于整个`Dense Block`，这个共享区也只需要分配 $WHk$（最大的`feature map` ）的内存，即内存消耗为$O(kL)$ (对比传统`DenseNet` 的 $O(kL^2)$)。

   - 后续的卷积操作直接从这个共享区读取数据。

   - 与`Concat` 操作共享区同样的原因，在反向传播阶段还需要重新计算第 层的`BN` 操作的结果。

     `BN` 的计算效率也很高，只需要额外付出大约 5% 的计算代价。

5. 由于`BN` 操作和`Concat` 操作在神经网络中大量使用，因此这种预分配共享内存区的方法可以广泛应用。它们可以在增加少量的计算时间的情况下节省大量的内存消耗。





## ShuffleNet

#### ShuffleNet

1. `ShuffleNet` 提出了 `1x1分组卷积+通道混洗` 的策略，在保证准确率的同时大幅降低计算成本。

   `ShuffleNet` 专为计算能力有限的设备（如：`10~150MFLOPs`）设计。在基于`ARM` 的移动设备上，`ShuffleNet` 与`AlexNet` 相比，在保持相当的准确率的同时，大约 13 倍的加速。

##### 9.3.1.1 ShuffleNet block

1. 在`Xception` 和`ResNeXt` 中，有大量的`1x1` 卷积，所以整体而言`1x1` 卷积的计算开销较大。如`ResNeXt` 的每个残差块中，`1x1` 卷积占据了`乘-加`运算的 93.4% （基数为32时）。

   在小型网络中，为了满足计算性能的约束（因为计算资源不够）需要控制计算量。虽然限制通道数量可以降低计算量，但这也可能会严重降低准确率。

   解决办法是：对`1x1` 卷积应用分组卷积，将每个 `1x1` 卷积仅仅在相应的通道分组上操作，这样就可以降低每个`1x1` 卷积的计算代价。

2. `1x1` 卷积仅在相应的通道分组上操作会带来一个副作用：每个通道的输出仅仅与该通道所在分组的输入（一般占总输入的比例较小）有关，与其它分组的输入（一般占总输入的比例较大）无关。这会阻止通道之间的信息流动，降低网络的表达能力。

   解决办法是：采用通道混洗，允许分组卷积从不同分组中获取输入。

   - 如下图所示：`(a)` 表示没有通道混洗的分组卷积；`(b)` 表示进行通道混洗的分组卷积；`(c)` 为`(b)` 的等效表示。
   - 由于通道混洗是可微的，因此它可以嵌入到网络中以进行端到端的训练。

   ![img](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shuffle_channel.png)

3. `ShuffleNet` 块的结构从`ResNeXt` 块改进而来：下图中`(a)` 是一个`ResNeXt` 块，`(b)` 是一个 `ShuffleNet` 块，`(c)` 是一个步长为`2` 的 `ShuffleNet` 块。

   在 `ShuffleNet` 块中：

   - 第一个`1x1` 卷积替换为`1x1` 分组卷积+通道随机混洗。

   - 第二个`1x1` 卷积替换为`1x1` 分组卷积，但是并没有附加通道随机混洗。这是为了简单起见，因为不附加通道随机混洗已经有了很好的结果。

   - 在`3x3 depthwise` 卷积之后只有`BN` 而没有`ReLU` 。

   - 当步长为2时：

     - 恒等映射直连替换为一个尺寸为 `3x3` 、步长为`2` 的平均池化。

     - `3x3 depthwise` 卷积的步长为`2` 。

     - 将残差部分与直连部分的`feature map` 拼接，而不是相加。

       因为当`feature map` 减半时，为了缓解信息丢失需要将输出通道数加倍从而保持模型的有效容量。

   ![img](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_block.png)

   

   小型网络通用设计准则

   ![image-20241027112332968](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20241027112332968.png)

   设计小型化网络目前有两个代表性的方向，并且都取得了成功：

- 通过一系列低秩卷积核（其中间输出采用非线性激活函数）去组合成一个线性卷积核或者非线性卷积核。如用`1x3 +3x1` 卷积去替代`3x3` 卷积 。
- 使用一系列稀疏卷积核去组成一个密集卷积核。如：交错卷积中采用一系列分组卷积去替代一个密集卷积。



##### ShuffleNet V2 block

1. `ShuffleNet V1 block` 的分组卷积违反了准则二，`1x1` 卷积违反了准则一，旁路连接的元素级加法违反了准则四。而`ShuffleNet V2 block` 修正了这些违背的地方。

2. `ShuffleNet V2 block` 在 `ShuffleNet V1 block` 的基础上修改。`(a),(b)` 表示`ShuffleNet V1 block` （步长分别为1、2），`(c),(d)` 表示`ShuffleNet V2 block` （步长分别为1、2）。其中`GConv` 表示分组卷积，`DWConv` 表示`depthwise` 卷积。

   - 当步长为1 时，`ShuffleNet V2 block` 首先将输入`feature map` 沿着通道进行拆分。设输入通道数为 ，则拆分为 和 。

     - 根据准则三，左路分支保持不变，右路分支执行各种卷积操作。

     - 根据准则一，右路的三个卷积操作都保持通道数不变。

     - 根据准则二，右路的两个`1x1` 卷积不再是分组卷积，而是标准的卷积操作。因为分组已经由通道拆分操作执行了。

     - 根据准则四，左右两路的`featuremap` 不再执行相加，而是执行特征拼接。

       可以将`Concat、Channel Shuffle、Channel Split` 融合到一个`element-wise` 操作中，这可以进一步降低`element-wise` 的操作数量。

   - 当步长为2时，`ShuffleNet V2 block` 不再拆分通道，因为通道数量需要翻倍从而保证模型的有效容量。

   - 在执行通道`Concat` 之后再进行通道混洗，这一点也与`ShuffleNet V1 block` 不同。

   ![img](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_v2_struct.png)









# 循环神经网络 RNN

1. 循环神经网络`recurrent neural network:RNN` ：用于处理序列数据 的神经网络，其中 表示第 个样本。 是一个序列 ，序列的长度可以是固定的、也可以是变化的。

   - 固定的序列长度：对每个样本 ，其序列长度都是常数 。
   - 可变的序列长度：样本 的序列长度 可能不等于样本 的序列长度 。

2. 循环神经网络是一种共享参数的网络：参数在每个时间点上共享。

   传统的前馈神经网络在每个时间点上分配一个独立的参数，因此网络需要学习每个时间点上的权重。而循环神经网络在每个时间点上共享相同的权重。

3. 循环网络中使用参数共享的前提是：相同参数可以用于不同的时间步。即：前一个时间步和后一个时间步之间的关系与时刻 无关。



rnn解决了一个什么问题？



1xn,nxn,nx1等等具体是什么？





![img](https://pic1.zhimg.com/v2-9a86430ba17aa299ce5c44c7b75c5ece_b.jpg)



X是一个向量，也就是某个字或词的特征向量，作为输入层，如上图也就是[3维向量](https://zhida.zhihu.com/search?content_id=115510247&content_type=Article&match_order=1&q=3维向量&zhida_source=entity)，

U是输入层到隐藏层的[参数矩阵](https://zhida.zhihu.com/search?content_id=115510247&content_type=Article&match_order=1&q=参数矩阵&zhida_source=entity)，在上图中其维度就是3X4，

S是隐藏层的向量，如上图维度就是4，

V是隐藏层到输出层的参数矩阵，在上图中就是4X2，

O是输出层的向量，在上图中维度为2。

![img](https://pic2.zhimg.com/v2-8abf977157000e6dad8589ec60ed6c3f_b.jpg)

举个例子，有一句话是，I love you，那么在利用RNN做一些事情时，比如命名实体识别，上图中的$X_{t-1}$代表的就是I这个单词的向量， X 代表的是love这个单词的向量， $X_{t+1}$ 代表的是you这个单词的向量，以此类推，我们注意到，上图展开后，W一直没有变，**W其实是每个时间点之间的[权重矩阵](https://zhida.zhihu.com/search?content_id=115510247&content_type=Article&match_order=1&q=权重矩阵&zhida_source=entity)**，我们注意到，RNN之所以可以解决序列问题，**是因为它可以记住每一时刻的信息，每一时刻的隐藏层不仅由该时刻的输入层决定，还由上一时刻的隐藏层决定**，公式如下，其中 $O_t$ 代表t时刻的输出, $S_t$ 代表t时刻的隐藏层的值：

![img](https://picx.zhimg.com/v2-9524a28210c98ed130644eb3c3002087_b.jpg)

**值得注意的一点是，在整个训练过程中，每一时刻所用的都是同样的W。**



## LSTM(Long Short-Term Memory)





![img](https://i-blog.csdnimg.cn/blog_migrate/a5f771bede44cfbee9cd9c694bd0a825.png)

<center class="half">
<img src="https://picx.zhimg.com/v2-5d50e60e1e4df95ca28da6a43c575713_b.jpg" alt="img" style="zoom:40%;" align = left />
<img src="https://pic4.zhimg.com/v2-712793fb8a9c7c08392bd3334f8fcb85_b.jpg" alt="img" style="zoom:33%;" align = right/>
<center>



 $ Z$ = $ \tanh $ (W[ $ x_ {t} $ , $ h_ {t-1} $ ])  ,   $ Z_ {i} $ = $ \sigma $ ( $ W_ {i} $ [ $ x_ {t} $ , $ h_ {t-1} $ ])
 $ Z_ {f} $ = $ \sigma $ ( $ W_ {f} $ [ $ x_ {t} $ , $ h_ {t-1} $ ])  ,    $ Z_ {0} $ = $ \sigma $ ( $ W_ {0} $ [ $ x_ {t} $ , $ h_ {t-1} $ ])

![img](https://i-blog.csdnimg.cn/blog_migrate/476d99bb79c51f9a29b00c2041c710d5.png)

## Transformer

### 重要组成部分

- 自注意力机制（Self-Attention）：这是Transformer的核心概念之一，它使模型能够同时考虑输入序列中的所有位置，而不是像循环神经网络（RNN）或卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中的不同部分来赋予不同的注意权重，从而更好地捕捉语义关系。
- 多头注意力（Multi-Head Attention）：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力允许模型并行处理不同的信息子空间。
- 堆叠层（Stacked Layers）：Transformer通常由多个相同的编码器和解码器层堆叠而成。这些堆叠的层有助于模型学习复杂的特征表示和语义。
- 位置编码（Positional Encoding）：由于Transformer没有内置的序列位置信息，它需要额外的位置编码来表达输入序列中单词的位置顺序。
- 残差连接和层归一化（Residual Connections and Layer Normalization）：这些技术有助于减轻训练过程中的梯度消失和爆炸问题，使模型更容易训练。
- 编码器（Encoder）和解码器（Decoder）：Transformer通常包括一个编码器用于处理输入序列和一个解码器用于生成输出序列，这使其适用于序列到序列的任务，如机器翻译。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3319e3d6922a2e7f2499a3130d3b5925.png#pic_center)



### Attention

1. 注意力函数可以描述为：将一个 `query` 和一组 `key-value pair` 映射到 `output` ，其中 `query, key, value, output` 都是向量。`output` 被计算为 `value` 的加权和，其中每个 `value` 的权重是由一个函数来计算（即，注意力函数），该函数以 `query, key` 作为输入。



> 注意力机制（attention）是一种可以为序列中的每个元素赋予权重的机制，该权重表示这个元素对其他元素的重要程度。自注意力机制（self-attention）是一种特殊的注意力机制，它能够在一个序列内部进行注意力计算。
>
> Transformer是一种使用自注意力机制来进行序列建模的深度学习模型。它在自然语言处理任务中取得了很大的成功，例如用于机器翻译的Transformer模型（Transformer Translation, 简称Transformer）。
>
> 以机器翻译任务为例，Transformer模型通过自注意力机制能够基于输入的源语言序列和目标语言序列来对每个序列位置进行自适应的注意力计算。这种注意力机制能够捕捉到不同位置上的词之间的依赖关系，并将这种依赖信息用于模型的建模和预测过程。
>
> Transformer模型通过多层的自注意力机制和前馈神经网络（feed-forward neural networks）来对输入序列进行编码，并对编码后的序列进行解码和预测。通过使用注意力机制，Transformer模型能够在更短的路径上捕捉到不同位置之间的依赖关系，使得模型更加高效且能够处理长序列，从而取得了很好的性能。

#### 自注意力机制



$Attention(Q,K,V) $ = $Softmax  (  \frac {QK^ {T}}{\sqrt {d_ {k}})}  ) \cdot V $





多头注意力

![image-20241105104138012](C:\Users\16277\AppData\Roaming\Typora\typora-user-images\image-20241105104138012.png)



`Multi-Head Attention`：与执行单个注意力函数 `attention function` （具有 $d_{model}$ 维的 `query, key, value` ）不同，我们发现将 `query, key, value` 线性投影 h 次是有益的，其中每次线性投影都是不同的并且将 `query, key, value` 分别投影到 $d_k, d_k, d_v$ 维。

然后，在每个 `query, key, value` 的投影后的版本上，我们并行执行注意力函数，每个注意力函数产生 $d_v$ 维的 `output` 。这些 `output` 被拼接起来并再次投影，产生 `final output` ，如上图右侧所示。

多头注意力`multi-head attention` 允许模型在每个`position` 联合地关注`jointly attend` 来自不同 `representation` 子空间的信息。如果只有单个注意力头`single attention head`，那么平均操作会抑制这一点。

$MultiHead(Q,K,V)$ =$ Concat (  \overrightarrow {head_ {1}}  ,  \overrightarrow {head_ {2}}  ,  \cdots  ,  \overrightarrow {head_ {h}}  ) \cdot W^O $


 $ \overrightarrow {head_ {i}} =Attention(  QW_ {i}^ {Q}  , KW_ {i}^ {K},  VW_ {i}^ {V} ) $





注意力在 `Transformer` 中的应用：`Transformer` 以三种不同的方式使用多头注意力：

- 在 `encoder-decoder attention` 层中，`query` 来自于前一个 `decoder layer`，`key` 和 `value` 来自于 `encoder` 的输出。这允许 `decoder` 中的每个位置关注 `input` 序列中的所有位置。这模仿了 `sequence-to-sequence` 模型中典型的 `encoder-decoder attention` 注意力机制。
- `encoder` 包含自注意力层。在自注意力层中，所有的 `query, key, value` 都来自于同一个地方（在这个 `case` 中，就是 `encoder` 中前一层的输出）。`encoder` 中的每个位置都可以关注 `encoder` 上一层中的所有位置。
- 类似地，`decoder` 中的自注意力层允许 `decoder` 中的每个位置关注 `decoder` 中截至到当前为止（包含当前位置）的所有位置。我们需要防止 `decoder` 中的信息向左流动，从而保持自回归特性。我们通过在 `scaled dot-product attention` 内部屏蔽掉 `softmax input` 的某些 `value` 来实现这一点（将这些 `value` 设置为 −∞ ），这些 `value` 对应于无效连接`illegal connection` 。

####  Why Self-Attention?

这里我们将自注意力层与循环层、卷积层进行各个方面的比较。这些层通常用于将一个可变长度的 `symbol representation` 序列 (x→1,⋯,x→n) 映射到另一个等长序列 (z→1,⋯,z→n)，其中 x→,z→i∈Rd 。 为了启发我们使用自注意力，我们考虑了三个方面：每层的总计算复杂度、可并行化的计算量（以所需的最少的序列操作数量来衡量）、网络中远程依赖`long-range dependency`的路径长度。

学习远程依赖是许多序列转导任务中的关键挑战。影响学习这种依赖的能力的一个关键因素是：前向传播信号和反向传播信号必须在网络中传输的路径长度。`input` 序列和 `output` 序列中任意位置组合之间的路径越短，那么就越容易学习远程依赖。因此，我们还比较了由不同类型的层组成的网络中，任意`input` 位置和 `output` 位置之间的最大路径长度。

如下表所示：

- 自注意力层在所有位置都关联一个 $O(1)$ 数量的序列操作（即具有最大程度的并行化），而循环层需要 $O(n)$ 数量的序列操作（即几乎完全无法并行化）。

  > 并行化指的是：为了计算指定位置的输出，模型需要依赖已经产生的多少个输出？

- 在计算复杂度方面，当序列长度$ n $小于 `representation` 维度 d 时（机器翻译中 `state-of-the-art` 模型常见的 `case` ），`self-attention` 层比循环层更快。

  为了提高涉及非常长序列（即 n 非常大）的任务的计算性能，可以将自注意力限制为仅考虑输入序列中的局部区域，这个局部区域是以输出位置对应处为中心、大小为 r 的邻域。这会将最大路径长度增加到$ O(n/r) $，同时每层的计算复杂度降低到 $O(rnd)$。我们计划在未来的工作中进一步研究这种方法。

- 具有 `kernel width` $k<n$ 的单个卷积层无法连接 `input position` 和 `output position` 组成的所有的 `pair` 。如果希望连接所有的 `input position` 和 `output position`，则需要堆叠 O(n/k) 个卷积层(连续核 `contiguous kernel`) 或 $O(log_k⁡(n))$ 个卷积层（空洞卷积 `dilated convolution` ），并且增加了网络中任意两个位置之间最长路径的长度。

  卷积层通常比循环层更昂贵 `expensive` （贵 k 倍）。然而，可分离卷积将计算复杂度显著降低到 $O(k×n×d+n×d2)$ 。当 k=n 时，可分离卷积的复杂度就等于一个自注意力层和一个`point-wise feed-forward layer` 的组合（这就是我们在 `Transformer` 模型中采用的方法）。



#### Decoder

和Encoder Block一样，Decoder也是由6个decoder堆叠而成的，Nx=6。包含两个 Multi-Head Attention 层。第一个 Multi-Head Attention 层采用了 Masked 操作。第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。



Masked Multi-Head Attention
与Encoder的Multi-Head Attention计算原理一样，只是多加了一个mask码。mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。为什么需要添加这两种mask码呢？

#### padding mask

什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。
具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！

#### sequence mask

sequence mask 是为了使得 decoder 不能看见未来的信息。对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。这在训练的时候有效，因为训练的时候每次我们是将target数据完整输入进decoder中地，预测时不需要，预测的时候我们只能得到前一时刻预测出的输出。
那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。
