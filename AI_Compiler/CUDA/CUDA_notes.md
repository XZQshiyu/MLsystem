## CUDA C++ Basics

Heterogeneous Computing

- Host: The CPU and its memory (host memory)
- Device: The GPU and its memory (device memory)

![image-20240828212339614](./assets/image-20240828212339614.png)

Simple processing flow

- Copy input data from CPU memory to GPU memory
- Load GPU program and execute caching data on chip for performance
- Copy results from GPU memory to GPU memory

![image-20240828213213016](./assets/image-20240828213213016.png)