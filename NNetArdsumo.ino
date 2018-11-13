/**
 * Example that drive Ardusumo in a obstacle avoidance task.
 * In "ArdusumoLib.h" are implemented simple routines to get infrared sensor
 * values normalized (0 to 1) and to set motor power by normalized 
 * values (-1 to 1)
 */


#include "NNet.h"
#include "ArdsumoLib.h"

float inp[2];      // buffer for sensor values
float out[2];      // buffer for motors power

char* netdef=                   //trained net definition
"L0 2 "                         //Layer 0 (input) of two nodes
"L1 3 NodeTanh "                //Layer 1 (hidden) of 3 nodes Tanh
"HLW0 -1.7421 1.8831 "          //Weights links to node 0 of hidden(L1) from L0
"HLW1 -1.2655 -1.2739 "         //         "            1        "        "
"HLW2 5.5744 5.5538 "           //         "            2        "        "
"L2 2 NodeTanh "                //Layer 2 (output) of 2 nodes Tanh
"OLW0 -1.0792 2.8989 2.8752 "   //Weights links to node 0 of output(L2) from L1
"OLW1 1.0836 3.2336 2.9449 "    //         "            1        "        " 
;

NNet net(netdef);               // net instance

void setup() {
  Serial.begin(9600);
  setupPins();
  delay(2000);
}

void loop() {
  delay(50);          // input->output step time (can be increased or decreased)
  usenet();
}


void usenet()
{
  inp[0]=ReadIrL();              // read left IR sensor
  inp[1]=ReadIrR();              // read right IR sensor 
  net.forw(inp,2,out,2);         // net execution
  MotorL(out[0]);                // apply power to left motor
  MotorR(out[1]);                // apply power to right motor 
}

