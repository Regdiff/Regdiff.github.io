安装MLIR

```
export PATH=/mnt/f/ubuntu_mlir/llvm-project/build/bin:$PATH
```









```
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja /mnt/f/ubuntu_mlir/llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```



1. Build Polygeist:

```
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/mnt/f/ubuntu_mlir/llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=/usr/bin/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
  
  
```







Hello MLIR
