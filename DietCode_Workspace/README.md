# DietCode

## Notes

- The page [here](https://github.com/Hzfengsy/tvm-tensorir/compare/main...dyn-dev) records the changes made.

- G4 Workstation

  ```Bash
  ssh -i G4Workstation.pem ubuntu@ec2-52-15-50-55.us-east-2.compute.amazonaws.com 
  ```

- Build Options:

  ```
  # CUTLASS
  cmake -DCUTLASS_NVCC_ARCHS=75 -DCMAKE_CUDA_ARCHITECTURES=75 .. # Tesla T4
  make -j 4 cutlass_lib
  ```
  ```
  # TVM
  cmake -DUSE_CUDA=/usr/local/cuda/ \
        -DUSE_LLVM=/usr/lib/llvm/bin/llvm-config \
        -DUSE_CUBLAS=1 \
        -DUSE_CUDNN=1 ..
  make -j 4
  ```

- Optimization Options:

  ```
  DIETCODE_SCHED_OPT                     Local Padding
  DIETCODE_SCHED_OPT_NO_LOCAL_PADDING    Disable local padding
  DIETCODE_SCHED_OPT_BLOCKIDX_PARTITION  blockIdx Partitioning
  ```
