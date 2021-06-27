### OpenCV compiled for Jetson with CUDA 

This directory contains  3 Python applications that use Yolo on OpenCv  with Python as programming language. These applications can run on Jetson or Windows

N.B. For better performance on Jetson OpenCv has to be compiled with CUDA option.

In this directory there is a link for downloading OpenCV, preconfigured for Jetson Nano and CUDA.  Downloading is external form Github because the file dimension (450MB).

To install OpenCV:

1. Download and decompress file in /home directory.
2. If not installed, install (`sudo apt-get install`) "libavcode, libavformat, libsvscale, libavutil e libdc1394"
3. Go to `/home/OpenCv-4.5.1/opencv-4.5.1/builder` . Then run `sudo make install`. If everithing is OK OpenCV is installed.

If installation don't succeed you have to recompile OpenCV, otherwise:

1. Go to `/home/OpenCv-4.5.1/opencv-4.5.1` 
2. Run `cmake-gui` (if not installed make`sudo apt-get install cmake-qt-gui`)
3. Cmake reads preconfigured “CmakeLists.txt”. Run *Configure* button and *Generate* button
4. If the output doesn't show severe error, you can proceed to the next point, else you have to modify compilation parameters or install everything  is missed, and repeat point 3.
5. Go, again, to `/home/OpenCv-4.5.1/opencv-4.5.1/build`
6. Run `nohup sudo make install -j4 &` and wait 2 hours more or less (note: nohup put compilation log on nohup.out and -j4  uses the 4 cores)

For speeding up compilation process is useful to enlarge swap file (virtual memory) before compiling:

1. Open with a text editor (in super user mode) the `/etc/dphys-swapfile` file
2. Make `CONF_SWAPSIZE = 2000` 

