# Artificial_Intelligence: Neural Networks
Neural Networks Library for Arduino.
This library is intended not just for Arduino but it can run also on PC,Rasperry Pi and so on.
If you want use library on different environment than Arduino you have to compile 2 files: "NNet.h" and "NNet.cpp" (or use these files on a IDE like "CodeBlocks").<br>
But, before compiling, you have to comment this line on "NNet.h":<br>
            <t>    #define ARDUINONNET    //uncomment if used on ARDUINO or comment if not<br>
As a matter of fact, this define disable some features not availlable on Arduino environment (as save net on file) but usefull in other contests.<br>
Library comes with a couple of examples and an help.
