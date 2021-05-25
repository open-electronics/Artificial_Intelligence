### Objects detecting with Yolo and Python 

This directory contains  3 Python applications that use Yolo on OpenCv  with Python as programming language. These applications can run on Jetson or Windows

N.B. For better performance on Jetson OpenCv has to be compiled with CUDA option.

- OpenCvYoloCamera : script to detect objects in real time using a webcam
- OpenCvYoloPhoto : script to describe objects in a photo
- OpenCvYoloAlarm : complete project for a intelligent alarm for detecting intruders

Directory darknet contain configuration and weights for some version of Yolo structures. It has to be positioned as described by  OpenCvYoloCamera and OpenCvYoloPhoto. 

Yolo networks are trained to recognize COCO 80 classes of objects, except yolo2-tiny-voc that is trained to recognize VOC 20 classes of object .

COCO labels (classes):

person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic, 
light, fire, hydrant, stop, sign, parking, meter, bench, bird, cat, dog, horse, 
sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, 
suitcase, frisbee, skis, snowboard, sports, ball, kite, baseball, bat, baseball, 
glove, skateboard, surfboard, tennis, racket, bottle, wine, glass, cup, fork, 
knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot, dog, 
pizza, donut, cake, chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, 
laptop, mouse, remote, keyboard, cell, phone, microwave, oven, toaster, sink, 
refrigerator, book, clock, vase, scissors, teddy, bear, hair, drier, toothbrush

VOC labels (classes)

aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, 
dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor