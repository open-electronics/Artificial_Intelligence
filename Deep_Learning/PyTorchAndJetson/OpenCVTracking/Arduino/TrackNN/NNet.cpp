/******************************************************************************
 NNetLib V3.0
 See NNet.h for more details.
 Copyright (C) Daniele Denaro (2019)
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.

*******************************************************************************/
#include "NNet.h"

typedef float (*pf)(float x);
pf getActFun(int f);

float fmax(float a,float b){return a>b?a:b ;}
float fmin(float a,float b){return a<b?a:b ;}
int imax(int a,int b){return a>b?a:b ;}
int imin(int a,int b){return a<b?a:b ;}

float Smaxfrw(float x);

float rnd(float a, float b)
{
    float r=rand();
    r=r/RAND_MAX;r= r * (b-a);r=r+a;
    return r;
}

float EPS=LCF;
float APH=DEC;
bool RMSprop=OptimizerRMSprop;

//#define eta 0.0001
#define eta 0.01



int state;
int node;
int counter;
/**********************************************************************/
void NNet::init(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type)
{
   lay0n=L0n;
   lay1n=L1n;
   lay2n=L2n;
   decodeType1(L1Type);
   decodeType2(L2Type);
   hidn=new Lhid[lay1n];
   outn=new Lout[lay2n];

   for (int i=0;i<lay1n;i++)
   {
       Lhid* hid=&hidn[i];
       hid->err=0;
       hid->out=0;
       hid->wgt=new Wgt[lay0n];
       for (int j=0;j<lay0n;j++){hid->wgt[j].w=rnd(RA,RB);hid->wgt[j].Gm2=0;}
   }

   for (int i=0;i<lay2n;i++)
   {
       Lout* out=&outn[i];
       out->out=0;
       out->wgt=new Wgt[lay1n];
       for (int j=0;j<lay1n;j++){out->wgt[j].w=rnd(RA,RB);out->wgt[j].Gm2=0;}
   }

   this->bhid=NULL;
   this->bout=NULL;
   this->hmem=NULL;
}

NNet::NNet(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type)
{
   init(L0n,L1n,L1Type,L2n,L2Type);
}

NNet::NNet(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type,bool bias,bool mem)
{
    init(L0n,L1n,L1Type,L2n,L2Type);
    if (bias)
    {
      bhid=new Wgt[lay1n];
      bout=new Wgt[lay2n];
      for (int j=0;j<lay1n;j++){bhid[j].w=rnd(RA,RB);bhid[j].Gm2=0;}
      for (int j=0;j<lay2n;j++){bout[j].w=rnd(RA,RB);bout[j].Gm2=0;}
    }
    if (mem)
    {
      hmem=new Hmem[lay1n];
      for (int i=0;i<lay1n;i++)
      {
          hmem[i].hm=0;
          hmem[i].wgt=new Wgt[lay1n];
          for(int j=0;j<lay1n;j++){hmem[i].wgt[j].w=rnd(RA,RB);hmem[i].wgt[j].Gm2=0;}
      }
    }
}

NNet::~NNet()
{
    for (int i=0;i<lay1n;i++) {delete[] hidn[i].wgt;}
    if (bhid!=NULL){delete[] bhid;}
    for (int i=0;i<lay2n;i++){delete[] outn[i].wgt;}
    if (bout!=NULL){delete[] bout;}
    if (hmem!=NULL) {for(int i=0;i<lay1n;i++){delete[] hmem[i].wgt;} delete[] hmem;}
    delete[] hidn;
    delete[] outn;
}

void NNet::forw(float inp[],float out[])
{
   float soutl1=0;
   float soutl2=0;

   for (int i=0;i<lay1n;i++)
   {
       float act=0;
       for (int k=0;k<lay0n;k++) {act=act+hidn[i].wgt[k].w*inp[k];}
       if(bhid!=NULL) act=act+bhid[i].w;
       if (hmem!=NULL)
        {for (int k=0;k<lay1n;k++) {act=act+hmem[i].wgt[k].w*hmem[k].hm;}}
       hidn[i].out=(*frw1)(act);
       hidn[i].err=0;
       soutl1=soutl1+hidn[i].out;
   }
   if (frw1 == Smaxfrw) {for (int i=0;i<lay1n;i++)hidn[i].out=hidn[i].out/soutl2;}

   for (int i=0;i<lay2n;i++)
   {
       float act=0;
       for (int k=0;k<lay1n;k++)
       {act=act+outn[i].wgt[k].w*hidn[k].out;}
       if(bout!=NULL) act=act+bout[i].w;
       outn[i].out=(*frw2)(act);
       soutl2=soutl2+outn[i].out;
   }
   if (frw2==Smaxfrw) {for (int i=0;i<lay2n;i++)outn[i].out=outn[i].out/soutl2;}

   for (int i=0;i<lay2n;i++) out[i]=outn[i].out;
   if(hmem!=NULL){for (int i=0;i<lay1n;i++)hmem[i].hm=hidn[i].out;}
}


float NNet::learn(float inp[],float train[])
{
   float out[lay2n];
   forw(inp,out);
   for (int i=0;i<lay2n;i++)
    {train[i]=-(out[i]-train[i]);}
   return backp(inp,train,false);
}

float NNet::learnpropagate(float inp[],float train[])
{
   float out[lay2n];
   forw(inp,out);
   for (int i=0;i<lay2n;i++)
    {train[i]=-(out[i]-train[i]);}
   return backp(inp,train,true);
}

float NNet::backlearn(float inp[],float bkerr[])
{
   return backp(inp,bkerr,true);
}

float NNet::backp(float inp[],float train[],bool bk)
{
    float errq=0;
    float errinp[1];
    if (bk){float errinp[lay0n];for(int i=0;i<lay0n;i++)errinp[i]=0;}
    for (int i=0;i<lay2n;i++)
    {
        float erri=train[i];
        errq=errq+(erri*erri);
        erri=erri*(*learn2)(outn[i].out);
        float errk,gd,gm2,dw;
        for (int k=0;k<lay1n;k++)
        {
           errk=erri * outn[i].wgt[k].w;
           hidn[k].err=hidn[k].err + errk;
           gd=erri*hidn[k].out;
           gm2=outn[i].wgt[k].Gm2;
           gm2=gm2+APH*(gd*gd-gm2);
           outn[i].wgt[k].Gm2=gm2;
           if(RMSprop)dw=EPS*gd/sqrt(gm2+eta);else dw=EPS*gd;
           outn[i].wgt[k].w=outn[i].wgt[k].w+dw;
        }
        if(bout!=NULL)
        {
            gd=erri;
            gm2=bout[i].Gm2;
            gm2=gm2+APH*(gd*gd-gm2);
            bout[i].Gm2=gm2;
            if(RMSprop)dw=EPS*gd/sqrt(gm2+eta);else dw=EPS*gd;
            bout[i].w=bout[i].w+dw;
        }
    }

    for (int i=0;i<lay1n;i++)
    {
        float erri=hidn[i].err;
        erri=erri*(*learn1)(hidn[i].out);
        float errk,gd,gm2,dw;
        for (int k=0;k<lay0n;k++)
        {
           errk=erri * hidn[i].wgt[k].w;
           if (bk){errinp[k]=errinp[k]+errk;}
           gd=erri*inp[k];
           gm2=hidn[i].wgt[k].Gm2;
           gm2=gm2+APH*(gd*gd-gm2);
           hidn[i].wgt[k].Gm2=gm2;
           if(RMSprop)dw=EPS*gd/sqrt(gm2+eta);else dw=EPS*gd;
           hidn[i].wgt[k].w=hidn[i].wgt[k].w+dw;
        }
        if(bhid!=NULL)
        {
            gd=erri;
            gm2=bhid[i].Gm2;
            gm2=gm2+APH*(gd*gd-gm2);
            bhid[i].Gm2=gm2;
            if(RMSprop)dw=EPS*gd/sqrt(gm2+eta);else dw=EPS*gd;
            bhid[i].w=bhid[i].w+dw;
        }

        if (hmem!=NULL)
        {
            for (int k=0;k<lay1n;k++)
            {
              gd=erri*hmem[k].hm;
              gm2=hmem[i].wgt[k].Gm2;
              gm2=gm2+APH*(gd*gd-gm2);
              hmem[i].wgt[k].Gm2=gm2;
              if(RMSprop)dw=EPS*gd/sqrt(gm2+eta);else dw=EPS*gd;
              hmem[i].wgt[k].w=hmem[i].wgt[k].w+dw;
            }
        }
    }
    if (bk){for(int i=0;i<lay0n;i++)inp[i]=errinp[i];}
    return errq/lay2n;
}

void NNet::resetMem()
{
    if (hmem==NULL) return;
    for (int i=0;i<lay1n;i++){hmem[i].hm=0;hidn[i].out=0;}
}

void NNet::resetLearnPar()
{
    for (int i=0;i<lay1n;i++)for(int j=0;j<lay0n;j++){hidn[i].wgt[j].Gm2=0;}
    if (bhid!=NULL){for (int i=0;i<lay1n;i++)bhid->Gm2=0;}
    for (int i=0;i<lay2n;i++)for(int j=0;j<lay1n;j++){outn[i].wgt[j].Gm2=0;}
    if (bout!=NULL){for (int i=0;i<lay2n;i++)bout->Gm2=0;}
    if (hmem!=NULL)
    {for (int i=0;i<lay1n;i++)for(int j=0;j<lay1n;j++){hmem[i].wgt[j].Gm2=0;}}
}

#ifdef ARDUINONNET
#include <ARDUINO.h>
void NNet::print()
{
    Serial.print("L0 ");Serial.print(lay0n);Serial.println();
    Serial.print("L1 ");Serial.print(lay1n);Serial.print(" ");Serial.print(ty1);Serial.println();
    if(bhid!=NULL){Serial.print("HLBS ");
                   for (int i=0;i<lay1n;i++){Serial.print(bhid[i].w,4);Serial.print(" ");}
                   Serial.println();}
    for (int i=0;i<lay1n;i++)
    {
        Serial.print("HLW");Serial.print(i);Serial.print(" ");

        for (int j=0;j<lay0n;j++)
        {Serial.print(hidn[i].wgt[j].w,4);Serial.print(" ");}
        Serial.println();
    }
    if (hmem!=NULL){
     for (int i=0;i<lay1n;i++)
     {
        Serial.print("HMW");Serial.print(i);Serial.print(" ");
        for (int j=0;j<lay0n;j++)
        {Serial.print(hmem[i].wgt[j].w);Serial.print(" ");}
        Serial.println();
     }
    }
    Serial.print("L2 ");Serial.print(lay2n);Serial.print(" ");Serial.print(ty2);Serial.println();
    if(bout!=NULL){Serial.print("OLBS ");
                   for (int i=0;i<lay1n;i++){Serial.print(bout[i].w,4);Serial.print(" ");}
                   Serial.println();}
    for (int i=0;i<lay2n;i++)
    {
        Serial.print("OLW");Serial.print(i);Serial.print(" ");
        for (int j=0;j<lay1n;j++)
        {Serial.print(outn[i].wgt[j].w,4);Serial.print(" ");}
        Serial.println();
    }
}

void serprintf(char* txt){Serial.print(txt);}
void serprintf(const char* txt,int v){char t[64];sprintf(t,txt,v);Serial.print(t);}
void serprintf(const char* txt,int v1,int v2){char t[64];sprintf(t,txt,v1,v2);Serial.print(t);}
void serprintf(const char* txt,float v){char t[12];dtostrf(v,5,4,t);Serial.print(t);}

void NNet::printPROGMEM()
{
    serprintf("const PROGMEM struct \n{\n");
    serprintf(" int dimin=%d;\n",lay0n);
    serprintf(" int dimhi=%d;\n",lay1n);
    serprintf(" int dimou=%d;\n",lay2n);
    serprintf(" int fun1=%d;\n",(int)cod1);
    serprintf(" int fun2=%d;\n",(int)cod2);

    serprintf(" float wgt10[%d][%d]=\n {\n",lay1n,lay0n);
    for (int y=0;y<lay1n;y++)
    {
        serprintf("  {");
        for (int x=0;x<lay0n;x++) {serprintf("%5.4f",hidn[y].wgt[x].w);if (x<lay0n-1) serprintf(", ");}
        serprintf("}");if (y<lay1n-1) serprintf(",\n"); else serprintf("\n");
    }
    serprintf(" };\n");

    serprintf(" float wgt21[%d][%d]=\n {\n",lay2n,lay1n);
    for (int z=0;z<lay2n;z++)
    {
        serprintf("  {");
        for (int y=0;y<lay1n;y++) {serprintf("%5.4f ",outn[z].wgt[y].w);if (y<lay1n-1) serprintf(", ");}
        serprintf("}");if (z<lay2n-1) serprintf(",\n"); else serprintf("\n");
    }
    serprintf(" };\n");

    if(bhid!=NULL){
         serprintf(" float bias1[%d]=\n {",lay1n);
         for (int y=0;y<lay1n;y++)
         {serprintf("%5.4f",bhid[y].w);if (y<lay1n-1) serprintf(", ");}
         serprintf("};\n");
    }
    if(bout!=NULL){
         serprintf(" float bias2[%d]=\n {",lay2n);
         for (int y=0;y<lay2n;y++)
         {serprintf("%5.4f",bout[y].w);if (y<lay2n-1) serprintf(", ");}
         serprintf("};\n");
    }

    if (hmem!=NULL){
    serprintf(" float whm11[%d][%d]=\n {\n",lay1n,lay1n);
    for (int y=0;y<lay1n;y++)
    {
        serprintf("  {");
        for (int x=0;x<lay1n;x++) {serprintf("%5.4f",hmem[y].wgt[x].w);if (x<lay1n-1) serprintf(", ");}
        serprintf("}");if (y<lay1n-1) serprintf(",\n"); else serprintf("\n");
    }
    serprintf("};\n");
    }
    serprintf("}pnet;");

}

/*
int d0,d1,d2;
pf frwd1,frwd2;
float *wgt10,*wgt21,*bias1,*bias2,*whm11,*hid,*hm;
bool bias,mem;
*/
struct NNPROGMEMpointers
{
  int d0,d1,d2;
  pf frwd1,frwd2;
  float *wgt10,*wgt21,*bias1,*bias2,*whm11,*hid,*hm;
  bool bias,mem;
};

NNPGM NNet::initNetPROGMEM(void* net,bool fbias,bool fmem)
{
  NNPGM pp=new NNPROGMEMpointers;
  pp->d0=pgm_read_word(net);
  pp->d1=pgm_read_word(net+2);
  pp->d2=pgm_read_word(net+4);
  pp->frwd1=getActFun(pgm_read_word(net+6));
  pp->frwd2=getActFun(pgm_read_word(net+8));
  pp->wgt10=(float*)(net+10);
  pp->wgt21=pp->wgt10+pp->d1*pp->d0;
  pp->bias1=pp->wgt21+pp->d2*pp->d1;
  pp->bias2=pp->bias1+pp->d1;
  pp->whm11=pp->wgt21+pp->d2*pp->d1;
  if (fbias) pp->whm11=pp->bias2+pp->d2;
  pp->hid=new float[pp->d1]; for (int y=0;y<pp->d1;y++){pp->hid[y]=0;}
  if (fmem) {pp->hm=new float[pp->d1];  for (int y=0;y<pp->d1;y++){pp->hm[y]=0;} }else pp->hm=NULL;
  pp->bias=fbias;
  pp->mem=fmem;
  return pp;
}

void NNet::forwPROGMEM(NNPGM pp,float inp[],float out[])
{
  float soutl1=0;
  float soutl2=0;

  for (int y=0;y<pp->d1;y++)
  {
      if (pp->bias) {pp->hid[y]=pgm_read_float(pp->bias1+y);} else pp->hid[y]=0;
      for (int x=0;x<pp->d0;x++)
      {pp->hid[y]=pp->hid[y]+inp[x]*pgm_read_float(pp->wgt10+y*pp->d0+x);}
      if (pp->mem){
       for (int h=0;h<pp->d1;h++)
       {pp->hid[y]=pp->hid[y]+pp->hm[h]*pgm_read_float(pp->whm11+y*pp->d1+h);}
      }
      pp->hid[y]=pp->frwd1(pp->hid[y]);
      soutl1=soutl1+pp->hid[y];
  }
  if (pp->frwd1 == Smaxfrw) {for (int i=0;i<pp->d1;i++)pp->hid[i]=pp->hid[i]/soutl2;}
  for (int z=0;z<pp->d2;z++)
  {
      if (pp->bias) out[z]=pgm_read_float(pp->bias2+z);else out[z]=0;
      for (int y=0;y<pp->d1;y++)
      {out[z]=out[z]+pp->hid[y]*pgm_read_float(pp->wgt21+z*pp->d1+y);}
      out[z]=pp->frwd2(out[z]);
      soutl2=soutl2+out[z];
  }
  if (pp->frwd2==Smaxfrw) {for (int i=0;i<pp->d2;i++)out[i]=out[i]/soutl2;}

  if (pp->mem)
  {for (int y=0;y<pp->d1;y++) pp->hm[y]=pp->hid[y];}
}



#else

void NNet::print()
{
    outdef(stdout,false);
}

void NNet::save(char* filename)
{
    FILE* fp;
    fp=fopen(filename,"w");
    if (fp==NULL) {printf("Can't open file: %s\n",filename);return;}
    outdef(fp,false);
    fclose(fp);
}

void NNet::savexardu(char* filename)
{
    FILE* fp;
    fp=fopen(filename,"w");
    if (fp==NULL) {printf("Can't open file: %s\n",filename);return;}
    outdef(fp,true);
    fclose(fp);
}

void NNet::savePROGMEM(char* filename)
{
    FILE* fp;
    fp=fopen(filename,"w");
    if (fp==NULL) {printf("Can't open file: %s\n",filename);return;}
    outdefPROGMEM(fp);
}


void NNet::outdef(FILE* fp,bool ardu)
{
    char q[3]="";char pc[3]="";
    if (ardu){strcpy(q,"\"");strcpy(pc,";");}
    fprintf(fp,"%sL0 %d %s\n",q,lay0n,q);
    fprintf(fp,"%sL1 %d %s %s\n",q,lay1n,ty1,q);
    if(bhid!=NULL){
        fprintf (fp,"%sHLBS ",q);
        for (int i=0;i<lay1n;i++)
        {fprintf(fp,"%6.4f ",bhid[i].w);}
        fprintf(fp,"%s\n",q);
    }
    for (int i=0;i<lay1n;i++)
    {
        fprintf (fp,"%sHLW%d ",q,i);
        for (int j=0;j<lay0n;j++)
        {fprintf(fp,"%6.4f ",hidn[i].wgt[j].w);}
        fprintf(fp,"%s\n",q);
    }
    if (hmem!=NULL){
     for (int i=0;i<lay1n;i++)
     {
        fprintf (fp,"%sHMW%d ",q,i);
        for (int j=0;j<lay1n;j++)
        {fprintf(fp,"%6.4f ",hmem[i].wgt[j].w);}
        fprintf(fp,"%s\n",q);
     }
    }

    fprintf(fp,"%sL2 %d %s %s\n",q,lay2n,ty2,q);

    if(bout!=NULL){
        fprintf (fp,"%sOLBS ",q);
        for (int i=0;i<lay2n;i++)
        {fprintf(fp,"%6.4f ",bout[i].w);}
        fprintf(fp,"%s\n",q);
    }
    for (int i=0;i<lay2n;i++)
    {
        fprintf (fp,"%sOLW%d ",q,i);
        for (int j=0;j<lay1n;j++)
        {fprintf(fp,"%6.4f ",outn[i].wgt[j].w);}
        fprintf(fp,"%s\n",q);
    }
    fprintf(fp,"%s",pc);
}

void NNet::outdefPROGMEM(FILE* fp)
{
    fprintf(fp,"const PROGMEM struct \n{\n");
    fprintf(fp," int dimin=%d;\n",lay0n);
    fprintf(fp," int dimhi=%d;\n",lay1n);
    fprintf(fp," int dimou=%d;\n",lay2n);
    fprintf(fp," int fun1=%d;\n",(int)cod1);
    fprintf(fp," int fun2=%d;\n",(int)cod2);

    fprintf(fp," float wgt10[%d][%d]=\n {\n",lay1n,lay0n);
    for (int y=0;y<lay1n;y++)
    {
        fprintf(fp,"  {");
        for (int x=0;x<lay0n;x++) {fprintf(fp,"%5.4f",hidn[y].wgt[x].w);if (x<lay0n-1) fprintf(fp,", ");}
        fprintf(fp,"}");if (y<lay1n-1) fprintf(fp,",\n"); else fprintf(fp,"\n");
    }
    fprintf(fp," };\n");

    fprintf(fp," float wgt21[%d][%d]=\n {\n",lay2n,lay1n);
    for (int z=0;z<lay2n;z++)
    {
        fprintf(fp,"  {");
        for (int y=0;y<lay1n;y++) {fprintf(fp,"%5.4f ",outn[z].wgt[y].w);if (y<lay1n-1) fprintf(fp,", ");}
        fprintf(fp,"}");if (z<lay2n-1) fprintf(fp,",\n"); else fprintf(fp,"\n");
    }
    fprintf(fp," };\n");

    if(bhid!=NULL){
         fprintf(fp," float bias1[%d]=\n {",lay1n);
         for (int y=0;y<lay1n;y++)
         {fprintf(fp,"%5.4f",bhid[y].w);if (y<lay1n-1) fprintf(fp,", ");}
         fprintf(fp,"};\n");
    }
    if(bout!=NULL){
         fprintf(fp," float bias2[%d]=\n {",lay2n);
         for (int y=0;y<lay2n;y++)
         {fprintf(fp,"%5.4f",bout[y].w);if (y<lay2n-1) fprintf(fp,", ");}
         fprintf(fp,"};\n");
    }

    if (hmem!=NULL){
    fprintf(fp," float whm11[%d][%d]=\n {\n",lay1n,lay1n);
    for (int y=0;y<lay1n;y++)
    {
        fprintf(fp,"  {");
        for (int x=0;x<lay1n;x++) {fprintf(fp,"%5.4f",hmem[y].wgt[x].w);if (x<lay1n-1) fprintf(fp,", ");}
        fprintf(fp,"}");if (y<lay1n-1) fprintf(fp,",\n"); else fprintf(fp,"\n");
    }
    fprintf(fp,"};\n");
    }
    fprintf(fp,"}pnet;");

}


NNet::NNet(const char* filename)
{
    FILE* fp;
    fp=fopen(filename,"r");
    if (fp==NULL) {printf("Can't open file: %s\n",filename);}
    init(0,0,"NodeLin",0,"NodeLin");
    char token[10]="";
    int err=0;
    state=0;node=0;counter=0;
    while (fscanf(fp,"%s",token)>0)
        {err=decode(token);if (err<0) return ;}
    return ;
}

#endif // NOARDUINO

NNet::NNet(char netdef[])
{
    init(0,0,"NodeLin",0,"NodeLin");
    char token[10]="";
    int err=0;
    state=0;node=0;counter=0;
    int p=0;
    int n=0;
    while (p<strlen(netdef))
       {sscanf(&netdef[p],"%s%*[' ']%n",token,&n);
        p=p+n;
        err=decode(token);if (err<0) return ;}
    resetLearnPar();
    return ;
}

int NNet::decode(char* token)
{
    if (strncasecmp(token,"L0",2)==0)  {newline(1,token);return 0;}
    if (strncasecmp(token,"L1",2)==0)  {newline(2,token);return 0;}
    if (strncasecmp(token,"HLBS",4)==0){newline(3,token);return 0;}
    if (strncasecmp(token,"HLW",3)==0) {newline(4,token);return 0;}
    if (strncasecmp(token,"L2",2)==0)  {newline(5,token);return 0;}
    if (strncasecmp(token,"OLBS",4)==0){newline(6,token);return 0;}
    if (strncasecmp(token,"HMW",3)==0) {newline(7,token);return 0;}
    if (strncasecmp(token,"OLW",3)==0) {newline(8,token);return 0;}
    tokenval(token);
}

void NNet::newline(int stat, char* token)
{
    state=stat;
    if (state==3) {bhid=new Wgt[lay1n];counter=0;return;}
    if (state==4) {counter=0;node=atoi(&token[3]);hidn[node].wgt=new Wgt[lay0n];return;}
    if (state==6) {bout=new Wgt[lay2n];counter=0;return;}
    if (state==7) {if (hmem==NULL)
                    {hmem=new Hmem[lay1n];
                     for(int i=0;i<lay1n;i++) {hmem[i].wgt=new Wgt[lay1n];hmem[i].hm=0;}}
                    counter=0;node=atoi(&token[3]);return;}
    if (state==8) {counter=0;node=atoi(&token[3]);outn[node].wgt=new Wgt[lay1n];return;}
}

void NNet::tokenval(char* token)
{
    switch (state)
    {
        case 1:lay0n=atoi(token);break;
        case 2:if (strncasecmp(token,"Node",4)==0) {decodeType1(token);break;}
               else {lay1n=atoi(token);hidn=new Lhid[lay1n];break;}
        case 3:bhid[counter].w=atof(token);counter++;break;
        case 4:hidn[node].wgt[counter].w=atof(token);counter++;break;
        case 5:if (strncasecmp(token,"Node",4)==0) {decodeType2(token);break;}
               else {lay2n=atoi(token);outn=new Lout[lay2n];break;}
        case 6:bout[counter].w=atof(token);counter++;break;
        case 7:hmem[node].wgt[counter].w=atof(token);counter++;break;
        case 8:outn[node].wgt[counter].w=atof(token);counter++;break;
    }
}


void NNet::setLearnCoeff(float cf){EPS=cf;}
void NNet::setRMSpDecay(float dec){APH=1-dec;}
void NNet::setRMSpOptimizer(bool y){RMSprop=y;}

void NNet::getHiddenOut(float buff[],int dimbuff)
{
    int k=imin(lay1n,dimbuff);
    for (int i=0;i<k;i++)buff[i]=hidn[i].out;
}

void NNet::getNetOut(float out[],int dimout)
{
    int k=imin(lay2n,dimout);
    for (int i=0;i<k;i++)out[i]=outn[i].out;
}

void NNet::getWeightsL1fromL0(int nodeL1,float w[],int dimw)
{
    int n=imin(nodeL1,lay1n);
    int k=imin(dimw,lay0n);
    for (int i;i<k;i++) w[i]=hidn[n].wgt[i].w;
}

void NNet::getWeightsL2fromL1(int nodeL2,float w[],int dimw)
{
    int n=imin(nodeL2,lay2n);
    int k=imin(dimw,lay1n);
    for (int i;i<k;i++) w[i]=outn[n].wgt[i].w;
}

int NNet::getnnodes(int layer)
{
    switch (layer)
    {
        case 0: return lay0n;
        case 1: return lay1n;
        case 2: return lay2n;
        default: return 0;
    }
}

/*********************************************************************/
#define NRELU 0.1

float linfrw(float x){return x;}
float linlrn(float y){return 1;}

float sigmfrw(float x){return 1/(1+exp(-x));}
float sigmlrn(float y){return y*(1-y);}

float tanhfrw(float x){float a=exp(x);float b=exp(-x);return (a-b)/(a+b);}
float tanhlrn(float y){return 1-(y*y);}

float ReLUfrw(float x){return x<0?NRELU*x:x ;}
float ReLUlrn(float y){return y<0?NRELU:1 ;}

float Smaxfrw(float x){return exp(x);}
float Smaxlrn(float y){return y*(1-y);}

typedef float (*pf)(float x);

pf getActFun(int f) //fun by index
{
  switch (f)
  {
    case 0: return linfrw;
    case 1: return sigmfrw;
    case 2: return tanhfrw;
    case 3: return ReLUfrw;
    case 4: return Smaxfrw;
    default: return linfrw;
  }
}

void decodefun(const char* ty,pf &fa,pf &fb,unsigned char &cod)
{
       if (strncasecmp(ty,"NodeLin",7)==0)  {fa=linfrw;fb=linlrn;cod=(unsigned char)0;}
       if (strncasecmp(ty,"NodeSigm",8)==0) {fa=sigmfrw;fb=sigmlrn;cod=(unsigned char)1;}
       if (strncasecmp(ty,"NodeTanh",8)==0) {fa=tanhfrw;fb=tanhlrn;cod=(unsigned char)2;}
       if (strncasecmp(ty,"NodeReLU",8)==0) {fa=ReLUfrw;fb=ReLUlrn;cod=(unsigned char)3;}
       if (strncasecmp(ty,"NodeSmax",8)==0) {fa=Smaxfrw;fb=Smaxlrn;cod=(unsigned char)4;}
}


void NNet::decodeType1(const char* ty)
{
    decodefun(ty,frw1,learn1,cod1);
    strncpy(ty1,ty,12);
}

void NNet::decodeType2(const char* ty)
{
    decodefun(ty,frw2,learn2,cod2);
    strncpy(ty2,ty,12);
}
