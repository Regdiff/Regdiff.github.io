
# 一个程序从编译到运行的全过程



### **前言**

一个程序，从编写完代码，到被计算机运行，总共需要经历以下四步：

1. **编译**。编译器会将[程序源代码](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=程序源代码&zhida_source=entity)编译成汇编代码。
2. **汇编**。汇编器会将汇编代码文件翻译成为[二进制](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=二进制&zhida_source=entity)的机器码。
3. **链接**。链接器会将一个个目标文件和库文件链接在一起，成为一个完整的[可执行程序](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=可执行程序&zhida_source=entity)。
4. **载入**。[加载器](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=加载器&zhida_source=entity)会将可执行文件的代码和数据从硬盘加载到内存中，然后跳转到程序的第一条指令处开始运行。



[链接器和加载器](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=链接器和加载器&zhida_source=entity)是由操作系统实现的程序。而编译器和[汇编器](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=2&q=汇编器&zhida_source=entity)则是由不同的编程语言自己实现的了。

这里需要展开来说一说，我们常用的高级语言，按照转化成机器码的方式不同可以分为**编译型语言和[解释型语言](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=解释型语言&zhida_source=entity)**，

- 编译型语言要求由**编译器**提前将源代码一次性转换成[二进制指令](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=二进制指令&zhida_source=entity)，即生成一个可执行程序，后续的执行无需重新编译。比如我们常见的 C、Golang 等，优点是执行效率高；缺点是可执行程序不能跨平台（不同的操作系统对不同的可执行文件的内部结构要求不同；另外，由于不同操作系统支持的函数等也可能不同，所以部分源代码也不能跨平台）。
- 解释型语言不需要提前编译，程序只在运行时才由**解释器**翻译成机器码，每执行依次就要翻译一次。比如我们常见的 Python、PHP 等，优点是较方便（对编写用户而言，省去了编译的步骤），实时性高（每次修改代码后都可直接运行），能跨平台；缺点是效率低。
- 半编译半解释型语言：还有一类比较特殊，混合了两种方式。源代码需要先**编译**成一种中间文件（[字节码文件](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=字节码文件&zhida_source=entity)），然后再将中间文件拿到虚拟机中**解释执行**。比如我们常见的 Java、C# 等。

所以，要设计一门语言，还必须为其编写相应的编译器和解释器，将源代码转化为计算机可执行的机器码。由于不同的语言有不同的转化方式，接下来将以最常见的 C 语言为例，简单分析一下 编译→汇编→链接→载入 的过程。

> 总结：不同的语言会使用不同的方式将源代码转化为机器码，但是之后的链接和载入过程都是由操作系统完成的，都是相同的。



### **1 编译**

编译是读取源程序，进行词法和[语法分析](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=语法分析&zhida_source=entity)，将高级语言代码转换为**汇编代码**。整个编译过程可以分为两个阶段。

### **1.1 预处理**

1. 对其中的[伪指令](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=伪指令&zhida_source=entity)（以 # 开头的指令）进行处理。

- 将所有的 `#define` 删除，并且展开所有的宏定义；
- 处理[条件编译](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=条件编译&zhida_source=entity)指令，如 `#if、#elif、#else、endif` 等；
- 处理[头文件](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=头文件&zhida_source=entity)包含指令，如 `#include`，将被包含的文件插入到该[预编译指令](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=预编译指令&zhida_source=entity)的位置；



1. 删除所有的注释。
2. 添加行号和文件名标识。

### **1.2 编译**

对预处理完的文件进行一系列[词法分析](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=词法分析&zhida_source=entity)、语法分析、语义分析及优化后，产生相应的汇编代码文件。



### **2 汇编**

将编译完的汇编代码文件翻译成**[机器指令](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=机器指令&zhida_source=entity)**，保存在后缀为 .o 的**目标文件**（Object File）中。

这个文件是一个 ELF 格式的文件（Executable and Linkable Format，可执行可链接文件格式），包括可以被执行的文件和可以被链接的文件（如目标文件 .o，可执行文件 .exe，[共享目标文件](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=共享目标文件&zhida_source=entity) .so），有其固定的格式。



### **3 链接**

由[汇编程序](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=汇编程序&zhida_source=entity)生成的目标文件并不能被立即执行，还需要通过链接器（Linker），将有关的目标文件彼此相连接，使得所有的目标文件成为一个能够被操作系统载入执行的统一整体。

> 例如在某个[源文件](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=源文件&zhida_source=entity)的函数中调用了另一个源文件中的函数；或者调用了库文件中的函数等等情况，都需要经过链接才能使用。

链接处理可以分为两种：

- **[静态链接](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=静态链接&zhida_source=entity)**：直接在编译阶段就把[静态库](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=静态库&zhida_source=entity)加入到可执行文件当中去。优点：不用担心目标用户缺少库文件。缺点：最终的可执行文件会较大；且多个应用程序之间无法共享库文件，会造成内存浪费。
- **[动态链接](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=动态链接&zhida_source=entity)**：在链接阶段只加入一些描述信息，等到程序执行时再从系统中把相应的动态库加载到内存中去。优点：可执行文件小；多个应用程序之间可以共享库文件。缺点：需要保证目标用户有相应的库文件。



### **4 载入**

加载器（Loader）会将可执行文件的代码和数据加载到内存（[虚拟内存](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=虚拟内存&zhida_source=entity)）中，然后跳转到程序的第一条指令开始执行程序。

说起来只是装载到内存中去那么简单的一句话，可是其实要是展开来说，会涉及到整个操作系统的内存管理。

### **虚拟内存**

首先，为了避免进程所使用的内存地址相互影响，操作系统会为每个进程分配一套**独立的虚拟内存地址**，然后再提供一种机制，**将虚拟内存地址和物理内存地址进行映射**。

- 我们程序所使用的内存地址叫做**虚拟内存地址**（Virtual Memory Address）
- 实际存在硬件里面的空间地址叫**物理内存地址**（Physical Memory Address）



### **用户空间**

然后，操作系统将整个内存空间分为**用户空间和[内核空间](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=内核空间&zhida_source=entity)**，其中内核空间只有内核程序能够访问，且所有进程共用一个内核空间；而[用户空间](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=3&q=用户空间&zhida_source=entity)是专门给应用程序使用的，每当创建了一个新的进程，都要分配一个用户空间。

![img](https://pic3.zhimg.com/v2-18eacffb74247975195b7af08b744348_b.jpg)

接下来以 32 位内存空间为例进行说明，32 位内存空间大小为 4GB，其中 1GB 为内核空间，3GB 为用户空间。用户空间中按照数据类型不同，划分为了不同的**内存段**，各类数据会被存放到各自的内存段中。

![img](https://pic1.zhimg.com/v2-3f2d3d755ffe4c9181c06d447001022a_b.jpg)

![img](https://picx.zhimg.com/v2-ce36fe48fe93bff0b881f3e0497a438d_b.jpg)

用户空间内存，从**低到高**分别是 6 种不同的内存段：

- 程序文件段（.text），包括**二进制[可执行代码](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=可执行代码&zhida_source=entity)**；
- 已初始化数据段（.data），包括**静态常量**；
- 未初始化数据段（.bss），包括**未初始化的[静态变量](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=静态变量&zhida_source=entity)**；
- 堆段，**包括动态分配的内存**，从低地址开始向上增长。当进程调用[malloc](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=malloc&zhida_source=entity)等函数分配内存时，新分配的内存就被动态添加到堆上（堆被扩张）；当利用free等函数释放内存时，被释放的内存从堆中被剔除（堆被缩减）；
- 文件映射段，包括动态库、共享内存等，从低地址开始向上增长；
- 栈段，包括**局部变量和函数调用的上下文**等。栈的大小是固定的，一般是 `8 MB`。当然系统也提供了参数，以便我们自定义大小；栈段可以通过[系统调用](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=系统调用&zhida_source=entity)自动地扩充空间，但是不能回收空间，所以栈段设置得太大会导致[内存泄露](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=内存泄露&zhida_source=entity)。



### **内存分页**

之后，操作系统**把整个虚拟和物理内存空间切成一段段固定尺寸的大小**。这样一个连续并且尺寸固定的内存空间，叫做**页**（Page）。在 Linux 下，每一页的大小为 `4KB`。

[虚拟地址](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=虚拟地址&zhida_source=entity)与物理地址之间通过**页表**来映射，CPU 中的 **MMU** （Memory Management Unit，[内存管理单元](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=内存管理单元&zhida_source=entity)）就做将虚拟地址转换成物理地址的工作。



### **总结**

至此，可执行文件载入内存的过程可以概括为以下几步：

1. 给进程分配虚拟内存空间；
2. 创建虚拟地址到物理地址的映射，创建页表；
3. 加载代码段和数据段等数据，即将硬盘中的文件拷贝到物理内存页中，并在页表中写入映射关系；
4. 把可执行文件的入口地址写入到 CPU 的 [指令寄存器](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=指令寄存器&zhida_source=entity)（PC）中，即可执行程序。



### **最后**

本文简单的描述了一下一个程序，从编写完代码，到被计算机运行的过程，其实当中的每一步都十分复杂深奥，都值得深入学习，尤其是最后一步载入内存的过程，展开来说可以涉及到整个操作系统的内存管理，像分段、分页、多级页表、TLB、内存分配、内存泄露、内存回收、[页面置换算法](https://zhida.zhihu.com/search?content_id=209813107&content_type=Article&match_order=1&q=页面置换算法&zhida_source=entity)等等都没能详细说，如果有机会的话我会慢慢补上的。如果你感兴趣的话可以留言催更，让我更有学习的动力！！！





之后，相关的机器码就会发出相应的控制信号。硬件就会愉快的跑起来程序啦。
