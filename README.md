# Neural Networks for Arduino (and for others micros)
NNlib has the ambition to make Neural Networks (NN for short) work on such reduced hardware as Arduino Uno. NNlib exists in two versions V3.0 and V2.5. First version V3.0 is more complete and it is planed for a wide range of hardware: from Arduino to PC. Second version V2.5 is calibrated particularly  for reduced hardware. This version is simplified but is slower because RAM optimization.
Both versions can run using PROGMEM structure for a previously trained NN. In this case only buffers for input, hidden nodes and output need to be automatically created because connection weights are in flash memory(V2.5 allocates just input an output buffers). Using this solution, even Arduino Uno can be used for a NN relatively large (for example: 256 input, 100 hidden, 10 out).

What is a Neural Network? There are many types of NN ; this library implements the feed-forward NN. A feed-forward NN can be seen as black-box that can be trained by examples.

![image1](/img/BlackBox.jpg)

A NN can be trained: to simulate any function (multidimensional) (in theory), to extract statistical behaviour, to categorize and so on. The sole training, by a big series of examples, is that you need. Of course, the chosen structure of NN can do training, more or less efficient. NNlib allows to define a feed-forward NN with two layer + input buffer. Dimension of input, hidden (intermediate) and output defines the NN behaviour.

![image2](/img/Strutture.jpg)

The library contains some examples of application. Also the well-known MNIST test is provided, but just for using on hardware environment with enough memory. The library is coded in C++. There are just two source files: **NNet.h** and **NNet.cpp** . You can use library moving it on Arduino IDE library folder, or you can use in another hardware/IDE compiling the library with your program.  If you use this library on different environment you have just to comment the “define” line about Arduino environment.
A detailed help is provided. 
