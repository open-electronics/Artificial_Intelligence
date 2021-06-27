### YOLO programs 

The directories Yolo-C, YoloPythonJetson e YoloPythonRasp, contain programs that use YOLO deep-learning convolutionary neural network (see [](https://pjreddie.com/darknet/yolo/)). This kind of deep-learning structure is able to detect and locate object in a frame.

Particularly:

- Yolo (C) : contains C programs (using darknet framework) for Windows and Raspberry Pi
- YoloPython : contains Python programs that use Opencv as framework to load and run YOLO trained network. This programs are trymed to use CUDA if present. So is useful on Jetson Nano  or Windows hardware.
- YoloPythonRasp : is a copy of previous programs but with a little customization for Raspberry Pi.

YoloPython contain also a complete application for a intelligent alarm that distinguish human body from other objects. 

