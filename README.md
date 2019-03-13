# Kmeans on ARM with SIMD
This is an implementation of a Naive K-Means implementation, using the MacQueeens algorithm on ARM. It attempts to use ARM Neon SIMD instructions where possible. This is done by detecting if the number of columns is a multiple of 16 (for 128-bit SIMD operations). If this is not possible, a check is then made for the number of columns being a (multiple of 2 for 64-bit SIMD operations). The fallback position is a CPU execution context, which will execute the Naive MacQueens algorithm on the CPU. 


## Options 
1) Any text file with data can be parsed to obtain the data to be characterised. The data is assumed to be in the 2-D format, with each row holding a data point. Each column holds the value of each dimension of the data. 
2) The initial starting centroids can be provided as a file in the same format as that of the data file or as an integer signifying the number of centroids. In the latter case, initial centroids are picked up as a pseudo random distribution of data points within the data sets. 
3) Data can initially be read in many different number formats. However this will eventually be typecast to float for calculation purposes. 
4) The Maximum number of iterations to converge on a solution can be passed in on the command line


## Build instructions
Build instructions assume that we are building for the Raspberry Pi 3. 
### Using a Cross compiler on a Linux host
1. Download source into \<source-dir\>
2. Obtain the Raspberry pi cross-compiler toolchain. Make sure you have write permissions to this directory. Use command 
   > git clone git@github.com:raspberrypi/tools.git
3. Create build and install directories separate from the source directory. 
4. Change to the build directory and execute the following CMake command : 
   > cmake -DTOOLCHAIN_DIR=\<toolchain-dir\> -DCMAKE_TOOLCHAIN_FILE=\<source-dir\>/toolchain-rpi.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=\<install-dir\>  \<source-dir\>
5. Build and install
   > make install


### Building on a Raspberry Pi
1. Download source into \<source-dir\>
2. Create build and install directories separate from the source directory. 
3. Change to the build directory and execute the following CMake command : 
   > cmake  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=\<install-dir\>  \<source-dir\>
4. Build and install
   > make install
