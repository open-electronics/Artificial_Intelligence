#include <iostream>

#include "../NNetLib30NoArdu/NNet.h"

using namespace std;

void readLab(char* filename);
void learn(int cycles,char* filename);
unsigned int readInt(FILE* fd);
void test(char* flab,char* fimg);
bool testout(int lab, float* out);

char* filelab="train-labels.idx1-ubyte";
char* fileimg="train-images.idx3-ubyte";

char* filelabT="t10k-labels.idx1-ubyte";
char* fileimgT="t10k-images.idx3-ubyte";

/* This version of main is just for testing.
*  If you want train your NN uncomment creation and learning section. And comment loading trained NN
*/

//NNet net(28*28,100,"NodeTanh",10,"NodeSmax");
NNet net("NetMNIST.txt");

int main()
{
    cout << "Test MNIST!" << endl;
//    readLab(filelab);
//    learn(100,fileimg);
//    net.save("NetMNIST.txt");
    test(filelabT,fileimgT);
    return 0;
}

unsigned char* lab;
int nlab;

unsigned char* buff;
float* inp;
float* trn;
float* out;

void readLab(char* filename)
{
   FILE* fd=NULL;
   fd=fopen(filename,"rb");
   if (fd==NULL){printf("Error file: %s\n",filelab);exit(-1);}
   int32_t mnuml;
   mnuml=readInt(fd);
   nlab=readInt(fd);
   printf("Labels magic:%i numimg:%i\n",mnuml,nlab);
   lab=new unsigned char[nlab];
   fread(lab,1,nlab,fd);
   fclose(fd);fd=NULL;
}
/*
void learn(int cycles,char* filename)
{
   net.setLearnCoeff(0.0001);
   FILE* fd=NULL;
   fd=fopen(filename,"rb");
   if (fd==NULL){printf("Error file: %s\n",fileimg);exit(-1);}

   FILE* fres=NULL;
   fres=fopen("Error.txt","w");
   if (fres==NULL){printf("Error file error\n");exit(-1);}

   int32_t mnumi,nimg,nr,nc;
   mnumi=readInt(fd);
   nimg=readInt(fd);
   nr=readInt(fd);
   nc=readInt(fd);
   printf("Imagins magic:%i numimg:%i\n",mnumi,nimg);
   int ninp=nr*nc;
   buff=new unsigned char[ninp];
   inp=new float[ninp];
   trn=new float[10];
   out=new float[10];
   int result;

   for (int k=0;k<cycles;k++)
   {
     result=0;
     float err=0;
     long pos=ftell(fd);
     for (int i=0;i<nimg;i++)
     {
       for (int n=0;n<10;n++){trn[n]=0;}
       trn[lab[i]]=1;
       fread(buff,1,ninp,fd);
       for (int x=0;x<ninp;x++) inp[x]=(float)buff[x]/255;
       err=err+net.learn(inp,trn);
       net.getNetOut(out,10);
       if (testout(lab[i],out)) result++;
     }
     printf("Cycle: %d  errqm: %7.6f  result: %4.1f %%\n",k,err/nimg,(float)(result*100)/nimg);
     fprintf(fres,"Cycle: %d  errqm: %7.6f  result: %4.1f %%\n",k,err/nimg,(float)(result*100)/nimg);
     fseek(fd,pos,SEEK_SET);
     net.save("NetMNIST.txt");
   }
   delete[] buff;delete[] inp;delete[] trn;delete[] out;delete[] lab;
}
*/
void test(char* flab,char* fimg)
{
   readLab(flab);

   FILE* fd=NULL;
   fd=fopen(fimg,"rb");
   if (fd==NULL){printf("Error file: %s\n",fileimg);exit(-1);}

   int32_t mnumi,nimg,nr,nc;
   mnumi=readInt(fd);
   nimg=readInt(fd);
   nr=readInt(fd);
   nc=readInt(fd);
   printf("Imagins magic:%i numimg:%i\n",mnumi,nimg);
   int ninp=nr*nc;
   buff=new unsigned char[ninp];
   inp=new float[ninp];
   float out[10];
   int labr=0;
   int result=0;
   for (int i=0;i<nimg;i++)
     {
       labr=lab[i];
       fread(buff,1,ninp,fd);
       for (int x=0;x<ninp;x++) inp[x]=(float)buff[x]/255;
       net.forw(inp,out);
       if (testout(labr,out)) result++;
     }
    printf("******************************\n");
    printf("Test result ok: %4.1f %%\n",(float)(result*100)/nimg);
}


unsigned int readInt(FILE* fd)
{
    unsigned char b[4];
    fread(b,1,4,fd);
    unsigned int ret=b[3]+(b[2]<<8)+(b[1]<<16)+(b[0]<<24);
    return ret;
}

bool testout(int lab, float* out)
{
    float maxout=0;
    int maxi=0;
    for (int y=0;y<10;y++) {if (out[y]>maxout) {maxout=out[y];maxi=y;}}
    return (maxi==lab);
}
