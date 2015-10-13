OpenCL Signed Distance Function
====

This is an OpenCL implementation that computes the exact
signed euclidean distance from binary images. 
The algorithm is from the paper "A Linear Time Algorithm for Computing
Exact Euclidean Distance Transforms of
Binary Images in Arbitrary Dimensions" by Maurer at al.

# Clone
This project uses a couple of wrappers and utils that are abailable
at the OZlib repository. In order for this submodules to be added
to your clone folder you need to do first your normal clonning:

    git clone git@github.com:olmozavala/2D_OpenCL_SDF.git

And then you add the submodules with:
    
    git submodule init

and download the latest code of the submodules with:

    git submodule update

# Build
This code has been tested with different flavors of Ubuntu and Nvidia cards. 
It uses the FreeImage library for image manipulation and premake4
to build the project. In ubuntu this two libraries can be installed with:

    sudo apt-get install premake4 libfreeimage3 libfreeimage-dev
    
Verify that the path of OPENCL in the 'premake4.lua' file
corresponds to the location of your opencl installation. In my case
it is set to '/usr/local/cuda'.

To compile the code you just need to run:
    premake4 gmake
    make
    make config=release ---> If you don't want debugging text

Or using the bash scripts:

    sh compile.sh

# Run
You can send as parameter one number, from 1 to 6, and it will select one of 
sample images in the images folder. Run the program with:

     ./dist/SignedDistFunc #

Or with the script file

    sh run.sh

The results are stored at the 'images' folder
