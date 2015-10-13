Created by Olmo Zavala-Romero
Florida State Unversity 18 Apr. 2012

This is an OpenCL implementation to compute the exact
signed euclidean distance from binary images. 
The algorithm is from the paper "A Linear Time Algorithm for Computing
Exact Euclidean Distance Transforms of
Binary Images in Arbitrary Dimensions" by Maurer at al.

I am still developing this code so you may encounter some small bugs. 

------- Required Software-----
This code has been tested with different flavors of Ubuntu and Nvidia cards. 
In order to compile you need to install:

CUDA ('Read CUDA_by_oz.txt for my own experience installing CUDA')
FreeImage --> libfreeimage-dev
premake4  --> premake4

------- Compile---------
To compile the code you just need to run:
premake4 gmake
make
make config=release ---> If you don't want debugging text

------- Run --------------
You can send as parameter the numbers 1-6 and it will select one of the images.

Run with:
./SignedDistFunc 1 
or
./SignedDistFunc 2 
...etc

The results are stored at the 'images' folder
