/**
 * Example that drive Ardusumo in a obstacle avoidance task.
 * Ardusumo is a rover with a couple of infrared sensors at +30 and -30 degree
 * in relation to forward direction. Sensor values are normalized to 1 
 * (from 0 to 1) and are approximately inversely propotional to distance 
 * (0<=4cm, 1>=40cm). Motion is provided by two motor (on diameter axis). Speed 
 * is normalized from -1 and 1.
 * This example can be easly customized for any other similar rover. 
 * In "ArdusumoLib.h" are implemented simple routines to get infrared sensor
 * values normalized (0 to 1) and to set motor power by normalized 
 * values (-1 to 1)
 * Two modes:
 * - using NN in PROGMEM : forward time 1.907 millisec
 * - using NN in RAM : forward time 1.900 millisec
 */


#include "NNet.h"
#include "ArdsumoLib.h"

float inp[2];      // buffer for sensor values
float out[2];      // buffer for motors power

const PROGMEM struct 
{
 int dimin=2;
 int dimhi=3;
 int dimou=2;
 int fun1=2;
 int fun2=2;
  float wgt10[3][2]=
  {
   {-1.7421, 1.8831},
   {-1.2655, -1.2739},
   {5.5744, 5.5538}
  };
  float wgt21[2][3]=
  {
   {-1.0792, 2.8989, 2.8752},
   {1.0836, 3.2336, 2.9449}
  };
 }pnet;

NNPGM pgm;   //pointer returned by initNetPROGMEM() function

void setup() {
  Serial.begin(9600);
  setupPins();
  pgm=NNet::initNetPROGMEM(&pnet,false,false);  // initialize NN in flash memory
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
  NNet::forwPROGMEM(pgm,inp,out); //net execution using PROGMEM data (1.907 millis)
  MotorL(out[0]);                // apply power to left motor
  MotorR(out[1]);                // apply power to right motor 
}

