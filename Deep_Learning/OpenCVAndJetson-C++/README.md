## Tracking using HAARCascade in C++

OpenCV library is a sophisticated functions container for images processing. Can be used with C++ language, Java or Python and can utilize CUDA accelerator if present. Unfortunately compiled version for Jetson Nano doesn't consider CUDA structure. So, it needs a complete compilation task. This task requires some hours. For this reason, a complete system image for Jetson Nano is provided in this repository. You can download this OS image and put it on a SD card.

OpenCV library includes a object detection using a HaarCascade method. This method is simpler than deep-learning convolution method but it is lighter and faster. More details can be found in 

https://docs.opencv.org/4.5.1/db/d28/tutorial_cascade_classifier.html

OpenCV version used in this example : OpenCV 4.5

In this directory we can see examples of object detecting using C++ in comparison with more useful Python program:

[]: ../PyTorchAndJetson/OpenCVTracking

These C++ example show the difference from Python version and C++ version.

Performance with C++ version

PC core i7 mem 16G  no CUDA:	about 30 frame/sec
Jetson NANO  no CUDA: 				about 7 frame/sec
Jetson NANO   CUDA:					  about 17 frame/sec

As you can see, with C++, the CUDA support on OpenCV for HaarCascade, is much efficient than using Python.

