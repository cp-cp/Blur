# Wave filtering
	受新兴的扩散模型（Diffusion Model）启发，
	思考到传统CPU滤波算法无法适应大模型时代的大数据量。
	考虑到使用并行计算的方法进一步提高算力，
	以训练更大的模型，形成新的商业模式。
	图像滤波器是为了解决噪点问题而建立的一个数学模型，通过这个模型来将图像数据进行能量转化。本项目基于C++/SYCL工具，实现对均值滤波与高斯滤波并行加速。
	包括并行计算下的高斯滤波实现，积分图优化的均值滤波

## 🔥 News

* `2023/07/15` 🤗🤗 We, as the participating project in the "[2023 Spring Intel oneAPI Campus Hackathon Competition](https://www.dev4one.com/web/season1#awards)" won the third prize and a special award, and received a 10,000 yuan reward from Intel Corporation. Thank you for Professor [Dai Hongju](https://faculty.sdu.edu.cn/daihongjun/zh_CN/index.htm)'s guidance.

## Prerequisites

| Optimized for | Description                      |
| ------------- | -------------------------------- |
| OS            | Ubuntu* 18.04                    |
| Software      | Intel® oneAPI DPC++/C++ Compiler |
| Dependency    | OpenCV                           |

## Build & Run

### On Linux

```sh
cmake .
make
./main {File_Name}
```

There are three sample picture in the folders. They are "**gaussian.png**", "**images.jpg**" and "**lena.png**".

You can use these command to take  a try.

 ```
 ./main gaussian.png
 ./main images.jpg
 ./main lena.png
 ```

The program will build three picture  "**gaussianOut.jpg**","**meanOut.jpg**" and "**meanPlusOut.jpg**".

### In Intel® DevCloud

you need to apply for Computing nodes with GPU.

Using the follow command, you will get into the  interactive mode.

```
qsub -I -l nodes=1:gpu:ppn=2 -d .
```

Then, you can use the same command **On Linux** to execute the program.

## Detail

You can see the detail how the program implements in the file "**文档.md**" in the repository.
