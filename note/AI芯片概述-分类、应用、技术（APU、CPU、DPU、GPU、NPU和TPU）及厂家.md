# AI芯片概述-分类、应用、技术（APU、CPU、DPU、GPU、NPU和TPU）及厂家





#### 文章目录

- [一、AI芯片是什么？](http://servicedev.tpddns.cn:8181/#AI_4)
- [二、AI芯片分类](http://servicedev.tpddns.cn:8181/#AI_8)
- - [1.Training(训练)](http://servicedev.tpddns.cn:8181/#1Training_12)
  - [2.Inference(推理)](http://servicedev.tpddns.cn:8181/#2Inference_14)
- [三、AI芯片应用领域](http://servicedev.tpddns.cn:8181/#AI_16)
- [四、AI芯片技术路线](http://servicedev.tpddns.cn:8181/#AI_30)
- [五、APU、CPU、DPU、GPU、NPU和TPU](http://servicedev.tpddns.cn:8181/#APUCPUDPUGPUNPUTPU_36)
- [六、AI芯片厂家](http://servicedev.tpddns.cn:8181/#AI_48)



## 一、AI芯片是什么？

**AI芯片**：针对人工智能算法做了特殊加速设计的芯片，也被称为**AI加速器**，即专门用于处理人工智能应用中的大量计算任务的功能模块。
**AI任务中99%以上的运算都是矩阵运算**。现在的AI模型中包含大量的矩阵运算，AI专用芯片都是对矩阵乘法做了优化，通过引入各种处理单元来高效进行矩阵运算。

> ```
> 本文章秉承对新手友好的原则，简单直白，如果有用请您点个关注鼓励一下，后续继续分享。
> ```

## 二、AI芯片分类

从业务应用来看，可以分为**Training(训练)**和**Inference(推理)**两个类型。
训练芯片需要考虑的因素更多，设计上也更加复杂，精度通常为FP32、FP16。
推理芯片考虑的因素较少，推理芯片考虑的因素较少，对精度要求也不高，INT8即可。

### 1.Training(训练)

Training环节通常需要训练出一个复杂的深度神经网络模型。训练过程涉及海量的训练数据和复杂的深度神经网络结构，运算量巨大，需要庞大的计算规模，对于处理器的计算能力、精度、可扩展性等性能要求很高。常用的例如华为的Atlas900集群、NVIDIA的GPU集群等。

### 2.Inference(推理)

Inference环节指利用训练好的模型，使用新的数据去“推理”出各种结论，如视频监控设备通过后台的深度神经网络模型，判断一张抓拍到的人脸是否属于特定的目标。虽然Inference的计算量相比Training少很多，但仍然涉及大量的矩阵运算。在推理环节，CPU、NPU、GPU和FPGA都有很多应用价值。

## 三、AI芯片应用领域

云端芯片部署在专业机房，对环境要求不高;边缘计算通常部署的户外，需要适应高温和低温环境;终端设备主要考虑功耗和成本。
**云端训练**
芯片特征：高功耗、高吞吐量、高精确率、分布式、可扩展性、高内存与带宽。
应用：云/HPC/数据中心
**云端推理**
芯片特征：低功耗、高吞吐量、高精确率、分布式、可扩展性、低延时。
应用：云/HPC/数据中心
**边缘计算**
芯片特征：低功耗、低延时，可单独部署或与其他设备组合、可将多个终端用户进行虚拟化、较小的机架空间。
应用：智能制造、智基家居智慧交通、智慧金融等众多领域。
**终端设备**
芯片特征：超低功耗、高能效、推理任务为主、较低的吞吐量、低延迟、成本敏感。
应用：各类消费电子产品形态多样、物联网领域。

## 四、AI芯片技术路线

目前这几个技术方向都在发力，不存在某个技术方向被淘汰。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/35516d319d3efe866400121d43ff9cee.png)
还有一些其他类型的AI芯片，如TPU（张量处理单元）、DPU（深度学习处理器）和VPU（视觉处理器）等，它们各自具有独特的设计和适用场景。后面有介绍。
FPGA与ASIC很多人可能不太熟悉，**FPGA**：Xilinx在芯片中增加HW/SW Programanle Engine，包含很多Al Core，Intel升级传统FPGA中的DSP模块。**ASIC**：各大公司都在布局ASIC，TPU、NPU、VPU等都属于ASIC。两者目前发力的领域不同，并无明显的竞争关系，对比如下。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/151af4664fdbc4a2299bf4cae3f5f0e7.png)

## 五、APU、CPU、DPU、GPU、NPU和TPU

总的来说，这些处理器各自具有独特的设计和功能，以满足不同的计算需求。APU融合了CPU和GPU的功能，CPU是通用处理器，DPU专注于数据处理，GPU适用于图形和图像计算，NPU专注于神经网络计算，而TPU则专门用于张量计算。在选择处理器时，需要根据具体的应用场景和需求进行考虑。以下是详细说明：
**APU（Accelerated Processing Units）**：APU是融合了CPU与GPU功能的产品，将两者的性能结合，实现最大性能。它结合了高性能处理器和最新独立显卡的处理性能，支持加速运算，从而大幅提升电脑的运行效率。APU适用于需要同时处理通用运算和大规模并行矢量运算的场景。
**CPU（Central Processing Unit）**：CPU是通用处理器，负责执行程序中的指令，进行顺序控制、操作控制、时间控制以及数据加工（算术运算和逻辑运算）。它是计算机系统的核心部件，适用于各种不同类型的计算任务。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/13aa3bffbea9b2e0792b7ec687dd3519.png)

**DPU（Data Processing Unit）**：DPU是由NVIDIA提出的一种专用处理器，以数据为中心构造。它负责处理CPU做不好、GPU做不了的数据任务，实现业务与基础设施的操作分离，提升性能，并提供零信任安全保护。DPU在数据中心和网络环境中有广泛应用。
**GPU（Graphics Processing Unit）**：GPU是一种专门用于处理图形和图像相关计算的硬件设备，具有高度并行的计算能力。它适用于处理大规模图形数据和复杂的计算任务，如科学计算、机器学习和人工智能等领域。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/376932ab3826c80b98ea7e450dcce95f.png)

**NPU（Neural Processing Unit）**：NPU是专门为神经网络计算设计的处理器，具有高效的神经网络运算能力。它可以改善机器学习的延迟、性能和能效比，并支持不同类型的神经网络模型。NPU在机器学习和人工智能领域有广泛应用。
**TPU（Tensor Processing Unit）**：TPU是由谷歌开发的专门用于进行张量计算的处理器，专注于高效执行大规模的张量计算，用于加速深度学习和人工智能任务。它在算力上具有优势，可以实现更高的计算效率和吞吐量。

## 六、AI芯片厂家

AI芯片厂家众多，以下是一些知名的厂家：
**英伟达（NVIDIA）**：作为全球GPU领域的领导者，英伟达在AI计算方面做出了巨大贡献。其创新的GPU技术不仅推动了图形处理的发展，还为深度学习等AI应用提供了强大的计算能力。
**英特尔（Intel）**：作为全球最大的芯片制造商之一，英特尔在AI芯片领域也有着深厚的积累。其AI芯片产品广泛应用于数据中心、自动驾驶等领域。
**AMD**：AMD也积极投入AI芯片的研发和生产，其产品在性能和能效比上表现出色，得到了市场的广泛认可。
**华为海思**：作为中国的代表性企业，华为海思在AI芯片领域也有着不俗的表现。其AI芯片产品具有高性能、低功耗等特点，广泛应用于智能手机、数据中心等领域。
**寒武纪**：寒武纪是中国的一家领先的AI芯片公司，其产品覆盖了云端、边缘端和终端等多个领域，为人工智能应用提供了强大的算力支持。
此外，还有一些其他的AI芯片厂家，如博通、阿斯麦、高通等，它们各自在AI芯片领域有着不同的优势和特点。

> ```
> 都看到这里了，请您点个关注鼓励一下，后续继续分享。
> ```