# 基础知识

### 内联函数

从作用上来讲，宏只是编译前的源文本替换，而内联函数是把函数要做的事直接展开在被调用处了。

运算之类的事情最好不要交给宏来做。

从引入意义上来讲，宏能让程序员偷点懒，内联函数能让程序少点负担。（当然如果调用处太多了最后出来的可执行文件就会比较大）



## 命名空间



`::` 用于明确指定命名空间中的成员，而 `using namespace` 则用于简化代码



```cpp
#include <iostream>

int x = 10;  // 全局变量 x

namespace MyNamespace {
    int x = 20;  // 命名空间 MyNamespace 中的变量 x
}

using namespace std;

int main() {
    cout << "Global x: " << ::x << endl;  // 使用全局变量 x
    cout << "MyNamespace x: " << MyNamespace::x << endl;  // 使用 MyNamespace 中的 x
    return 0;
}
```



## 动态内存

了解动态内存在 C++ 中是如何工作的是成为一名合格的 C++ 程序员必不可少的。C++ 程序中的内存分为两个部分：

- **栈：**在函数内部声明的所有变量都将占用栈内存。
- **堆：**这是程序中未使用的内存，在程序运行时可用于动态分配内存。

很多时候，您无法提前预知需要多少内存来存储某个定义变量中的特定信息，所需内存的大小需要在运行时才能确定。

## 堆和栈

在计算机体系结构中，栈（stack）和堆（heap）是两种不同的内存区域，它们在内存管理、分配方式、使用场景以及生命周期等方面都有显著的区别。下面是栈和堆的主要区别：

### 栈 (Stack)

1. **分配与释放**：
   - **自动分配与释放**：栈上的内存分配和释放是自动进行的。当一个函数被调用时，其局部变量和函数参数会被推入栈中；当函数返回时，这些变量会被自动从栈中弹出。
   - **后进先出 (LIFO)**：栈遵循后进先出的原则，最后进入栈的数据最先被移除。

2. **大小限制**：
   - **固定大小**：栈的大小通常是固定的，并且相对较小。栈的大小通常在程序编译时确定，由操作系统或运行时环境配置。
   - **栈溢出**：如果递归调用过深或者局部变量占用过多内存，可能会导致栈溢出（stack overflow）。

3. **访问速度**：
   - **快速访问**：由于栈的简单结构和自动管理，栈上的内存访问非常快。

4. **用途**：
   - **局部变量**：主要用于存储函数的局部变量、函数参数以及返回地址等。
   - **递归**：递归函数的调用依赖于栈来保存每一层的函数状态。

5. **碎片问题**：
   - **无碎片**：由于栈的 LIFO 特性，不会产生内存碎片。

### 堆 (Heap)

1. **分配与释放**：
   - **手动分配与释放**：堆上的内存分配和释放需要程序员显式地进行。在C++中，可以使用 `new` 和 `delete` 来分配和释放堆内存。
   - **动态分配**：堆内存可以在程序运行时动态分配和释放。

2. **大小限制**：
   - **可变大小**：堆的大小通常是可变的，取决于可用的系统内存。堆的大小比栈大得多，可以用来存储大量的数据。

3. **访问速度**：
   - **较慢访问**：由于堆的复杂管理和动态分配特性，堆上的内存访问速度相对较慢。

4. **用途**：
   - **动态数据结构**：主要用于存储动态分配的数据结构，如链表、树、图等。
   - **全局对象**：可以用来存储全局对象或需要跨多个函数调用保持存在的对象。

5. **碎片问题**：
   - **可能产生碎片**：频繁的分配和释放可能导致堆内存碎片化，影响内存使用效率。现代内存管理器通常会有一些策略来减少碎片的影响。

### 总结

- **栈** 是一个自动管理的、固定大小的内存区域，用于存储局部变量和函数调用信息，访问速度快，但大小有限。
- **堆** 是一个手动管理的、可变大小的内存区域，用于动态分配内存，适用于大型数据结构和跨函数调用的数据，但访问速度较慢，且可能产生内存碎片。

理解栈和堆的区别对于编写高效且安全的代码非常重要，特别是在处理大量数据或长时间运行的应用程序时。

## new和malloc的区别

在C++中，`new` 和 `malloc` 都可以用来动态分配内存，但它们之间有一些重要的区别。这些区别主要体现在以下几个方面：

### 1. 语言支持
- **`new`** 是C++中的运算符，专门设计用于对象的动态内存分配。
- **`malloc`** 是C语言中的函数，也可以在C++中使用，但它主要用于基本类型的内存分配。

### 2. 类型安全性
- **`new`** 是类型安全的。它知道要分配内存的对象类型，并返回适当类型的指针。例如：
  ```cpp
  int* p = new int;  // 分配一个int类型的内存，并返回int*指针
  ```
- **`malloc`** 不是类型安全的。它需要你手动指定要分配的字节数，并返回一个 `void*` 指针，你需要将其转换为适当的类型。例如：
  ```cpp
  int* p = (int*)malloc(sizeof(int));  // 分配sizeof(int)个字节，并返回void*指针
  ```

### 3. 构造和析构
- **`new`** 在分配内存后会调用对象的构造函数来初始化对象。
- **`malloc`** 只分配内存，不会调用任何构造函数。你需要自己进行初始化。

### 4. 内存对齐
- **`new`** 自动处理内存对齐问题，确保分配的内存适合特定类型的对象。
- **`malloc`** 不保证内存对齐，但在大多数现代系统上通常也会正确对齐。

### 5. 异常处理
- **`new`** 在内存分配失败时会抛出 `std::bad_alloc` 异常。
- **`malloc`** 在内存分配失败时返回 `NULL`。

### 6. 释放内存
- **`new`** 分配的内存需要用 `delete` 来释放。
  ```cpp
  delete p;  // 释放由new分配的内存，并调用析构函数
  ```
- **`malloc`** 分配的内存需要用 `free` 来释放。
  ```cpp
  free(p);  // 释放由malloc分配的内存
  ```

### 7. 数组分配
- **`new`** 可以用来分配数组，并且可以自动计算数组元素的数量。
  ```cpp
  int* arr = new int[10];  // 分配一个包含10个int的数组
  delete[] arr;  // 释放数组
  ```
- **`malloc`** 也可以用来分配数组，但你需要手动计算所需的字节数。
  ```cpp
  int* arr = (int*)malloc(10 * sizeof(int));  // 分配一个包含10个int的数组
  free(arr);  // 释放数组
  ```

### 示例代码

#### 使用 `new`
```cpp
#include <iostream>

int main() {
    int* p = new int(42);  // 分配一个int，并初始化为42
    std::cout << *p << std::endl;
    delete p;  // 释放内存
    return 0;
}
```

#### 使用 `malloc`
```cpp
#include <iostream>
#include <cstdlib>  // 包含malloc和free

int main() {
    int* p = (int*)malloc(sizeof(int));  // 分配一个int的内存
    if (p != NULL) {
        *p = 42;  // 手动初始化
        std::cout << *p << std::endl;
        free(p);  // 释放内存
    } else {
        std::cerr << "Memory allocation failed" << std::endl;
    }
    return 0;
}
```

### 总结
- **`new`** 是C++中推荐使用的动态内存分配方式，因为它提供了类型安全、自动构造和析构等功能。
- **`malloc`** 是C语言中的内存分配函数，在C++中也可以使用，但它缺乏类型安全性和自动构造/析构的功能。

如果你在编写C++代码，建议优先使用 `new` 和 `delete`。如果你需要与C代码兼容或有特殊需求，可以使用 `malloc` 和 `free`。

## 指针

### base

```
char * char_pt = & a;
int * dp = & var1[0];   
```

&是取地址

*是

1. **指针声明**： 当 `*` 出现在变量声明旁边时，它表示该变量是一个指针。指针变量存储的是另一个变量的内存地址。例如：

   ```
   int x = 10;
   int *ptr = &x; // ptr 是一个指向整数的指针，并被初始化为 x 的地址
   ```

2. **解引用运算符**： 当 `*` 应用于一个指针时，它用来访问指针所指向的对象。这个过程叫做“解引用”。例如：

   ```
   int value = 20;
   int *p = &value; // p 指向 value
   cout << *p; // 输出 20，因为 *p 访问了 p 所指向的值
   ```



无论一个指针的类型是什么，他们总是同一种数据结构，那就是地址。

指针也可以++，--，当然，他变化的值基本上都是四个字节，无论是指向字节还是int

```
-----------------------------
before : 0x62fdf0
after "dp++" 0x62fdf4
```





### 结构体

在C语言中，结构体（struct）是一种用户自定义的数据类型，它允许将不同类型的数据组合在一起。当你定义了一个结构体变量后，你可以使用点运算符 `.` 来访问该结构体的成员。

这个结构体的数据结构叫做**type_name**，我们创建了两个具体的数据为**type_name1**，和**type_name2**

例如：
```cpp
struct type_name {
    member_type name;
    member_type age;
    member_type length;
} type_name1, type_name2;

// 访问type_name1的name成员
type_name1.name = some_value;  // 使用点运算符
```

然而，当你通过指针来访问结构体变量时，你需要使用箭头运算符 `->`。这是因为指针存储的是结构体变量的地址，而不是实际的结构体数据。为了访问指针所指向的结构体中的成员，你需要先解引用这个指针，然后才能访问到结构体成员。

例如：
```cpp
struct type_name *ok = &type_name1;  // ok是指向type_name1的指针

// 访问ok所指向的age成员
ok->age = some_other_value;  // 使用箭头运算符
```

箭头运算符 `->` 实际上是两个操作的结合：首先对左边的指针进行解引用，然后对解引用后的结果应用点运算符。上面的例子等价于：
```cpp
(*ok).age = some_other_value;  // 显式地解引用和使用点运算符
```

总结一下：
- 当你直接使用结构体变量名时，用 `.` 来访问其成员。
- 当你使用指向结构体的指针时，用 `->` 来访问被指向结构体的成员。

### 传递指针给函数

```
void increment(int *num) {
    (*num)++;  // 通过指针解引用修改传入的整数值
}
```

还有一个点就是，void increment(int *num) 中 num已经定义了，不需要在定义了



### C++ 从函数返回指针

```
#include <iostream>

int* createArray(int size) {
    int *arr = new int[size];  // 动态分配一个整数数组
    for (int i = 0; i < size; ++i) {
        arr[i] = i;  // 初始化数组元素
    }
    return arr;  // 返回指向数组的指针
}

int main() {
    int size = 5;  // 我们想要创建一个包含5个元素的数组
    int *array = createArray(size);  // 调用函数并接收返回的指针
}
```



###  指向类的指针



**动态分配内存**

指向类的指针还可以用于动态分配内存，创建类的对象：



```c++

class MyClass {
public:
  int data;
  void display() {  
  }
};

int main() {
  // 动态分配内存创建类对象
  MyClass *ptr = new MyClass;
  ptr->data = 42;

  // 通过指针调用成员函数
  ptr->display();

  // 释放动态分配的内存
  delete ptr;

  return 0;
}
```

# 数据结构

## vector

## class



## 数组

```
//创建一维静态数组
void CreatArray01()
{
	const int N = 100;
	//int arr[]={1,2,3,4,5,6,7,8,9};//可以直接赋值
	int Array01[N] = { 0 };	//可以将数组中所有元素赋值为0，其他值不可以这样操作
	for (int i = 0; i < N; i++)	cin >> Array01[i];
}

//用new创建一维动态数组
void CreatArray02()
{
	int num;	//表示数组元素的数量
	cin >> num;
	int* Array02 = new int[num];
	for (int i = 0; i < num; i++)	cin >> Array02[i];
}

//用vector创建一维数组
void CreatArray03() {
    int num;
    cin >> num;
    
    vector<int> Array03(num);
 
    for (int i = 0; i < num; i++) {
        cin >> Array03[i];
    }
}
```



### 指针数组

```
VertexNode * AdjList[MaxVertices]; 
// 这行代码声明了一个数组 AdjList，其中每个元素是一个指向 VertexNode 类型的指针。
// 这种数据结构通常用于表示图（Graph）的邻接表（Adjacency List）。
```



## 链表

单项链表

```
struct node
{
	int data;        //数据
	node* next;      //指向下一个结点的指针
};
```

## 有向图

```cpp
#include <bits/stdc++.h>
using namespace std;
#define MaxVertices 100

//定义结点
struct VertexNode {
	int data;	//结点编号
	int weight = 0;	//指向下一个结点的边的权值
	VertexNode *next = NULL;
};

//定义邻接表
struct GraphAdjList {
	VertexNode *AdjList[MaxVertices];	//存储所有结点
	int numV, numE;
};

//创建图
void CreatGraph(GraphAdjList &G) {
	int vi, vj, w;
	cout << "请输入顶点数：" << endl;
	cin >> G.numV;
	cout << "请输入顶点信息：" << endl;
	for (int i = 0; i < G.numV; i++) {
		cin >> vi;
		VertexNode *new_node = new VertexNode;
		new_node->data = vi;
		G.AdjList[i] = new_node;
	}
	cout << "请输入边的数量：" << endl;
	cin >> G.numE;
	cout << "请输入边的信息：" << endl;
	for (int i = 0; i < G.numE; i++) {
		cin >> vi >> vj >> w;
		//找到邻接表中对应结点的位置，往其中链表插入对应边
		for (int j = 0; j < G.numV; j++) {
			if (vi == G.AdjList[j]->data) {
				VertexNode *temp = G.AdjList[j];
				//这里用的是尾插法
				while (temp->next != NULL) {
					temp = temp->next;
				}
				VertexNode *newEdge = new VertexNode;
				newEdge->data = vj;
				newEdge->weight = w;
				temp->next = newEdge;
				break;
			}
		}
	}
}

//遍历图
void showGraph(GraphAdjList &G) {
	for (int i = 0; i < G.numV; i++) {
		VertexNode *temp = G.AdjList[i]->next;
		int vi = G.AdjList[i]->data;
		cout << "顶点" << vi << "的边有：" << endl;
		if (temp == NULL) {
			cout << "无" << endl;
		}
		while (temp != NULL) {
			cout << vi << "->" << temp->data << " 权值=" << temp->weight << endl;
			temp = temp->next;
		}
	}
}

int main() {
	GraphAdjList GAL;
	CreatGraph(GAL);
	showGraph(GAL);
}

```



# 算法

## DP

### [零钱兑换](https://leetcode.cn/problems/coin-change/)





给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。

你可以认为每种硬币的数量是无限的。

 

**示例 1：**

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

**示例 2：**

```
输入：coins = [2], amount = 3
输出：-1
```

**示例 3：**

```
输入：coins
```

Answer:

```cpp
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int Max = amount + 1;
        vector<int> dp(amount+1, Max);
        dp[0] = 0;
        for (int i = 1 ; i < amount+1 ; i ++){
            for (int j = 0 ; j < coins.size() ; j ++){
                if (i >= coins[j])
                    dp[i] = min(dp[i], dp[i-coins[j]] + 1);
            }

        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
};
```

