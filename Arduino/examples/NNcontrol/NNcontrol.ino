/* This example realizes a closed loop control system (like PID)
*  For this purpose a simple NN (2,2,1) has been trained to emulate a surface
*  (f(E,EC)) produced by a Fuzzy controller (see MatLab) where E is the error 
*  between Reference and Output of controlled system and EC its change at each 
*  step (a sort of derivative).
*  Training of this NN is a regression task, in other words a function 
*  approssimation. Of course surface is not linear and this feature is the power 
*  of this kind of fuzzy control system. But using NN is faster than using 
*  inference engine in fuzzy system and it is more accurate than a look-up table
*  of surface values.
*  This fuzzy controller is normalized from -1 and 1 both for E and EC, but also
*  for control output value.
*  Can be used for controlling different system just appling a coefficient Kp 
*  anf Kd to input E and EC. Another output coefficient can be applied to scale
*  or reverse control signal.
*  
*  Two modes:
*  - Using NN in PROGMEM : each execution time = 0.508 millisec
*  - Using NN in RAM : each execution time = 0.500 millisec
*/


#include "NNet.h"

#define useprogmem  //comment if you prefere NN in RAM

#ifdef useprogmem
 const PROGMEM struct 
 {
  int dimin=2;
  int dimhi=2;
  int dimou=1;
  int fun1=2;
  int fun2=0;
  float wgt10[2][2]=
  {{0.0031, -2.0110},{-2.0180, 0.0037}};
  float wgt21[1][2]=
  {{-0.5035 , -0.5028 }};
 }pnet;
#else
 char* netdef=
 "L0 2 "
 "L1 2 NodeTanh "
 "HLW0 0.0031 -2.0110 "
 "HLW1 -2.0180 0.0037 "
 "L2 1 NodeLin "
 "OLW0 -0.5035 -0.5028 "
 ;
 
 NNet net(netdef);
#endif

float inp[2];

NNPGM p;    //p is a pointer to one initialized pnet (different p points to different pnet)  

void setup() {
   Serial.begin(9600);
#ifdef useprogmem   
   p=NNet::initNetPROGMEM(&pnet,false,false);//with V3.0 you have to inizialize
#endif   
/* just for example and timing. Cancel for your application  */ 
   unsigned long ta,tb;
   ta=millis();
   for(int i=0;i<1000;i++) control(inp);
   tb=millis();
   Serial.print("Time: ");Serial.println(tb-ta);
/**/   
}

void loop() {
   //put here your control 
}

float control(float inp[2])
{
  float out[1];
#ifdef useprogmem
  NNet::forwPROGMEM(p,inp,out);          // execution time = 0.520 millis
#else
  net. forw(inp,out);                    // execution time = 0.520 millis
#endif
  return out[0];   
}

/* Example of complete control routine */
float control(float x,float dx,float kp, float ki, float kd)
{
   
}

