Normalize

为什么要进行Normalize呢？
在神经网络进行训练之前，都需要对于输入数据进行Normalize归一化，目的有二：1，能够加快训练的速度。2.提高训练的稳定性。

为什么使用Layer Normalization（LN）而不使用Batch Normalization（BN）呢？


先看图，LN是在同一个样本中不同神经元之间进行归一化，而BN是在同一个batch中不同样本之间的同一位置的神经元之间进行归一化。
BN是对于相同的维度进行归一化，但是咱们NLP中输入的都是词向量，一个300维的词向量，单独去分析它的每一维是没有意义地，在每一维上进行归一化也是适合地，因此这里选用的是LN。

————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。

原文链接：https://blog.csdn.net/Tink1995/article/details/105080033



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