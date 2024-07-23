## [有没有一本讲解gpu和CUDA编程的经典入门书籍？](https://www.zhihu.com/question/26570985/answer/3559066509)



光看书肯定不够，比如目前没有什么书讲tensor core的优化，这部分实际应用非常广泛。1，看视频，迅速了解 GPU的基础知识，了解 GPU 编程模型、显存、[共享内存](https://www.zhihu.com/search?q=共享内存&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3559066509})等；[NVIDIA CUDA初级教程视频_哔哩哔哩_bilibiliwww.bilibili.com/video/BV1kx411m7Fk/?spm_id_from=333.337.search-card.all.click&vd_source=0d656ebac9f6ff3cdced8f1272b08374![img](https://pica.zhimg.com/v2-b7ac5e9a2cc9a4d2c57df51327e0843a_ipico.jpg)](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1kx411m7Fk/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D0d656ebac9f6ff3cdced8f1272b08374)这个视频虽然老，但是对于这些基础知识，非常重要；

2，看书 [通用图形处理器](https://www.zhihu.com/search?q=通用图形处理器&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3559066509})设计:GPGPU[编程模型与架构原理](https://www.zhihu.com/search?q=编程模型与架构原理&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3559066509}) ，更了解GPU架构，打下基础；

3，实际应用课程，里面有主流的FlashAttention、 RingAttention 、 triton等最新的以gpu为中心的大模型技术；[https://github.com/cuda-modegithub.com/cuda-mode](https://link.zhihu.com/?target=https%3A//github.com/cuda-mode)

4，自己写算子，如简单的融合算子、softmax、gemv、简单的[矩阵乘法](https://www.zhihu.com/search?q=矩阵乘法&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3559066509})、rms_norm、用triton写一个FlashAttention，写着写着就发现，cuda学了一点，也写了一个简单版本的大模型[推理引擎](https://www.zhihu.com/search?q=推理引擎&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3559066509})，关键是这些算子网上也有很多讲解，不会写也可以学习。