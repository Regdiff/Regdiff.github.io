***第三代框架***:

我们除了设计框架解决当前的问题，还应该思考关注和设计下一代的框架以支持未来的模型趋势。

[![img](https://github.com/microsoft/AI-System/raw/main/Textbook/%E7%AC%AC1%E7%AB%A0-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%B3%BB%E7%BB%9F%E6%A6%82%E8%BF%B0/img/2/2-3-3-framework-2to3.png)](https://github.com/microsoft/AI-System/blob/main/Textbook/第1章-人工智能系统概述/img/2/2-3-3-framework-2to3.png)

图 1.2.3 第二代框架到第三代框架的发展趋势

- 框架应在有更加全面功能的编程语言前端下构建，并提供灵活性和表达力，例如：控制流（Control Flow）的支持，递归和稀疏性的原生表达与支持。这样才能应对大的（Large）、动态（Dynamic）的和自我修改（Self-Modifying）的深度学习模型趋势。我们无法准确预估深度学习模型在多年后会是什么样子，但从现在的趋势看，它们将会更大、更稀疏、结构更松散。下一代框架应该更好地支持像 [Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)[[20\]](https://github.com/microsoft/AI-System/blob/main/Textbook/第1章-人工智能系统概述/1.2-算法，框架，体系结构与算力的进步.md#pathways) 模型这样的动态模型，像预训练神经语言模型（NLM）或多专家混合模型（MoE）这样的大型模型，以及需要与真实或模拟环境频繁交互的强化学习模型等多样的需求。
- 框架同时应该不断跟进并提供针对多样且新的硬件特性下的编译优化与运行时调度的优化支持。例如：单指令流多数据流（SIMD）到 多指令流多数据流（MIMD）的支持，稀疏性和量化的硬件内支持，异构与分布式计算，虚拟化支持，关联处理等。





