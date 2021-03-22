/*
* Tracking with 2 servos (Pan, Tilt) drived by a Neural Network acting as Fuzzy Controller. 
* Servo 1 on pin 9
* Servo 2 on pin 10
* Center coordinates from computer camera; coordinates recived by USB serial port.
* Format of commands (characters) via serial port (9600 bauds) :
*                      0.nnn 0.nnn /n 
* where 0.nnn:x coordinate and 0.nnn:y coordinate
*  or
*                          ? /n 
* just replays with sketch name itsMe() function
*/
#include <Servo.h>  // servo library
#define SRVPIN1 9   // pin for servo 1
#define SRVPIN2 10  // pin for servo 2
#define bauds 57600  // serial speed

#include "NNet.h"   // Neural Networks library for Arduino

/* Neural Network definition */
char* netdef=
"L0 2 "                      // input layer: 2 nodes
"L1 2 NodeTanh "             // first layer: 2 nodes with Tanh as activation function
"HLW0 0.0031 -2.0110 "       // weights from input layer to first layer (to node 0)
"HLW1 -2.0180 0.0037 "       //    "                                    (to node 1) 
"L2 1 NodeLin "              // output layer: 1 node with Linear activation function
"OLW0 -0.5035 -0.5028 "      // weights from first layer to output node
;

NNet nncontrol(netdef);      // NN instance

char EL=10;       // line-feed
char CR=13;        // cariage return
char buff[64];    // receiving buffer 

Servo sw1;        // two servo object (Pan)
Servo sw2;        // two servo object (Tilt)

float val1,val2;  // buffer for input fron serial

float g=1;        // scale factor for controller output (degree variation)
float df=1;
float a=1;
float b=1;

boolean fservo=true;
 
//constraints for pan and tilt
float minx=20;
float maxx=160;
float miny=20;
float maxy=160;

//buffers
float ex,ey,dex,dey;
float oldex,oldey;
float dax, day;
float ix,iy;

//angles (degree)
float angx=90;
float angy=90;

void setup() {
  sw1.attach(SRVPIN1);   // init servo 1
  sw2.attach(SRVPIN2);   // init servo 2
  reset();               // initial position 90° 90°
  Serial.begin(bauds);   // init serial
  Serial.println("Starting...");
}

void loop() 
{
  if (Serial.available()>0)     // if something in buffer read it
  {                         
    int n=Serial.readBytesUntil(EL,buff,32); // read record
    buff[n]=0;
    if(n>0) decode();
  }
}

void decode()                   // decode first character or basic command
{
  if (buff[0]=='?') {itsMe();return;}       // display sketch name
  if (buff[0]=='r') {reset();return;}       // reset buffers
  if (buff[0]=='s') {servonoff();return;}   // on/off servos
// basic command: face positions x y
  char* p;                 
  p=strtok(buff," ");val1=(float)atof(p);     // x (centre)
  p=strtok(NULL," ");val2=(float)atof(p);     // y (centre)
  NNControl(val1,val2);                       // NN controller
}

void itsMe()                    // Routine to identify sketch in memory
{                               // It replies with file name without extension
  char name[]=__FILE__;
  char* c=strchr(name,'.');
  if (c != NULL) *c='\0';
  Serial.println(name);
}

void reset()
{
  angx=sw1.read();
  angy=sw2.read();
  moveToAngle(90, 90);
  ex=0;ey=0;oldex=0;oldey=0; 
}

void servonoff()
{
  if (fservo) fservo=false;
  else fservo=true;
}

//buffer for input-output controller
float binx[2];   // from -1 to 1
float boux[1];   // from -1 to 1
float biny[2];   // from -1 to 1
float bouy[1];   // from -1 to 1

void NNControl(float x,float y) 
//x,y center normalized coordinates (0 to 1)
{
  ix=(x-0.5)*1;  //from -1 to 1
  iy=(y-0.5)*1;  //from -1 to 1
  ex=ix;
  ey=iy;
  dex=(ex-oldex)*2; //amplify variations
  dey=(ey-oldey)*2;
  binx[0]=ex;
  binx[1]=dex;
  nncontrol.forw(binx,boux); 
  biny[0]=ey;
  biny[1]=dey;
  nncontrol.forw(biny,bouy); 
  float ax=-boux[0]*g; 
  float ay=-bouy[0]*g; 
  dax=ax;
  day=ay; 
  cmdSetAngle(dax,day);
  oldex=ex;
  oldey=ey;
  float repv[]={x,y,ex,dex,ey,dey,dax,day,angx,angy};
  reply(repv,10);
}

void cmdSetAngle(float gx,float gy)
{
  angx=checkMarginx(angx+gx);
  angy=checkMarginy(angy+gy);
  if (fservo)
  {
   sw1.write(round(angx));             // pan
   sw2.write(round(angy));             // tilt
  } 
}

void moveToAngle(int ax,int ay)
{
  int newx=checkMarginx(ax);
  int newy=checkMarginy(ay);
  int da=1;
  for (int i=0;i<100;i++)
  { 
   if ((abs(angx-newx)<=da)&(abs(angy-newy)<=da)) break;
   if (angx<newx)  {angx=angx+da; sw1.write(angx);}             // pan
   if (angx>newx)  {angx=angx-da; sw1.write(angx);}             // pan
   if (angy<newy)  {angy=angy+da; sw2.write(angy);}             // tilt  
   if (angy>newy)  {angy=angy-da; sw2.write(angy);}             // tilt
   delay(3);
  } 
}

float checkMarginx(float gx)
{return constrain(gx,minx,maxx);}

float checkMarginy(float gy)
{return constrain(gy,miny,maxy);}

void reply(float v[],int n)
{
  for (int i=0;i<n;i++)
   {
    Serial.print(v[i]);
    Serial.print(" ");
   }
   Serial.println();
}


