MLIR

MLIR和TVM间的关系是什么？TVM我理解是解决了不同的上层框架以及不同的AI芯片之间的对接问题，目的是使Framework不需要针对每一个AI芯片都开发一套适配代码。MLIR我感觉也是提供的一层这种编译上的抽象，通过Dialect可以做到高层IR到低层IR之间的转换，是不是功能有一定的重合？如果要开发一款新的AI芯片，需要提供自己的编译器，那如何使用MLIR？是需要进行各种AI框架（TF/Pytorch/......），通过MLIR Dialect转换成此AI芯片的IR表达吗？MLIR的Pass中，是不是只能进行与具体硬件无关的High Level的优化，而不支持与具体硬件相关的Low Level的优化？这部分优化需要AI芯片自己的编译器实现？





上一篇文章讲到当前的编译结构的问题在于各种IR之间转换的效率和可迁移性不高。MLIR试图使用一种一致性强的方式，为各种DSL提供一种中间表达形式，将他们集成为一套生态系统，编译到特定硬件平台的汇编语言上。这样的目标是通过什么手段实现的呢？





> 如果基于MLIR的compiler如何支持分布式计算呢？还是说分布式计算本身不应该是compiler考虑的东西





https://github.com/GaoXiangYa/mlir-tutorial-ch/blob/main/README.md