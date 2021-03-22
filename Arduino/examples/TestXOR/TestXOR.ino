/* 
    Example: training and use of XOR function (classical basic example like 
   "Hello word" in computer language programming explanation)
   Two phases switched by "#define training" and "#define deploiment" statements
   - 1 Training using a new NN definitition. Ending with printPROGMEN().
   - 2 Copying struct into flash memory and use this net for testing results
*/   

#include "NNet.h"

/* uncomment one of next define in order to use learning or deploiment phase */

//#define training
#define deploiment

#ifdef deploiment
const PROGMEM struct 
{
 int dimin=2;
 int dimhi=2;
 int dimou=1;
 int fun1=2;
 int fun2=0;
 float wgt10[2][2]=
 {
  {-0.4758, -0.4790},
  {-2.6100, -2.6207}
 };
 float wgt21[1][2]=
 {
  {3.3193, -2.4683}
 };
}pnet;
#endif

#ifdef training 
/*net definition 2 node input,2 node hidden Tanh and 1 node out linear */
NNet net(2,2,"NodeTanh",1,"NodeLin");  
#endif  

void setup() {
  Serial.begin(9600);
  Serial.print("Test NNet train V30\n");
  int samples=5000;
  Serial.print("NNet train for ");Serial.print(samples);Serial.println(" samples...");
#ifdef training  
  testxor(samples);
  net.print();
  net.printPROGMEM(); //print in structure format (select on console and use ctr-c to copy)
#endif
#ifdef deploiment  
  showXorFun(); 
#endif  
}

void loop() {}

float fxor[4][3]={{0,0,0},{0,1,1},{1,0,1},{1,1,0}};   // XOR function

char rec[80];

void testxor(int samples)
{
    float vin[2];
    float vout[1];
    float vtr[1];
    float errq=0;
    int k=samples/10;
    char sfloat[12];
    for (int i=1;i<samples;i++)
    {
        int r=random(0,4);
        vin[0]=fxor[r][0];
        vin[1]=fxor[r][1];
        vtr[0]=fxor[r][2];
#ifdef training        
        errq=errq+net.learn(vin,vtr);
#endif        
        dtostrf(errq,6,5,sfloat);
        if (i%k==0) 
         {errq=errq/k;sprintf(rec,"%d Errq: %s \n",i,sfloat);Serial.print(rec);errq=0;showXorFun();}
    }
}

void showXorFun()
{
    float vin[2];
    float vout[1];
    float trn;
    int err=0;
    char sfloat[12];
    NNPGM np;

    for(int i=0;i<4;i++)
    {
        vin[0]=fxor[i][0];
        vin[1]=fxor[i][1];
        trn=fxor[i][2];
#ifdef training        
        net.forw(vin,vout);
#endif
#ifdef deploiment 
        np=NNet::initNetPROGMEM(&pnet,false,false); //with V3.0 you need to initialize PROGMEM net       
        NNet::forwPROGMEM(np,vin,vout);             //before using it
#endif        
        int in0= vin[0]>0.5?1:0;
        int in1= vin[1]>0.5?1:0;
        int ou= vout[0]>0.5?1:0;
        err=err+fabs(trn-ou);
        dtostrf(vout[0],6,5,sfloat);
        sprintf(rec,"     %d %d -> %d  (%s)\n",in0,in1,ou,sfloat);
        Serial.print(rec);
    }
}

