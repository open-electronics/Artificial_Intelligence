### Python programs with YOLO and OpenCv 

This directory is equivalent to the directory for Jetson, but programs are trimmed for Raspberry Pi.

In "OpenCV4Raspberry" directory there is a link for downloading OpenCV, preconfigured for Raspberry Pi.  Downloading is external form Github because the file dimension (450MB).

To install OpenCV:

1. Download and decompress file in /home directory.
2. If not installed, install (`sudo apt-get install`) "libavcode, libavformat, libsvscale, libavutil e libdc1394"
3. Go to `/home/OpenCv-4.5.2/opencv-4.5.2/build` . Then run `sudo make install`. If everithing is OK OpenCV is installed.

If installation don't succeed you have to recompile OpenCV, otherwise:

1. Go to `/home/OpenCv-4.5.2/opencv-4.5.2` 
2. Run `cmake-gui` (if not installed make`sudo apt-get install cmake-qt-gui`)
3. Cmake reads preconfigured “CmakeLists.txt”. Run *Configure* button and *Generate* button
4. If the output doesn't show severe error, you can proceed to the next point, else you have to modify compilation parameters or install everything  is missed, and repeat point 3.
5. Go, again, to `/home/OpenCv-4.5.2/opencv-4.5.2/build`
6. Run `nohup sudo make install -j4 &` and wait 2 hours more or less (note: nohup put compilation log on nohup.out and -j4  uses the 4 cores)

If Raspberry Pi has 8GB of memory compilation process should not have particularly problems. On different situation is useful to enlarge swap file (virtual memory) before compiling:

1. Open with a text editor (in super user mode) the `/etc/dphys-swapfile` file
2. Make `CONF_SWAPSIZE = 2000` (default `CONF_SWAPSIZE = 100`)



When OpenCv is installed you can test these programs that use YOLO deep-learning network for object detecting.

- OpenCvYoloAlarm  is a complete application for implementing a intelligent alarm ( see doc inside)
- OpenCvYoloCamera try to detect objects in real time
- OpenCvYoloPhoto allows you to choose photos and analyze it

