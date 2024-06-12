### 内联函数

从作用上来讲，宏只是编译前的源文本替换，而内联函数是把函数要做的事直接展开在被调用处了。

运算之类的事情最好不要交给宏来做。

从引入意义上来讲，宏能让程序员偷点懒，内联函数能让程序少点负担。（当然如果调用处太多了最后出来的[可执行文件](https://www.zhihu.com/search?q=可执行文件&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"1050972991"})就会比较大）



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



