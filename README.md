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