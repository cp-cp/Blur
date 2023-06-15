## Prerequisites

| Optimized for | Description                      |
| ------------- | -------------------------------- |
| OS            | Ubuntu* 18.04                    |
| Software      | Intel® oneAPI DPC++/C++ Compiler |
| Dependency    | OpenCV                           |



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

You can see the detail how the program implements in the file "**文档.md**" in the repository.