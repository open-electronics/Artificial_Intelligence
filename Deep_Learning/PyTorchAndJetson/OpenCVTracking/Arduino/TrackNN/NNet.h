/**************************************************************************************
Version V3.0
Neural Networks library. It can works also on Arduino platform.
If it runs on Arduino you have to uncomment "#define ARDUINONNET"
By contrast, if library is used on computer, you have to comment this define statement if you want
use useful functions as save net on file or load net from file, available only on computer or on
Arduino with SD memory card.

You can use Neural Network of 3 layer.
Each layer can have any number of nodes.
Layer 0 is the input layer (just to collect input values); transfer function is linear.
Layer 1 is the hidden layer. Each node summarize inputs weighted by input links.
For this layer you can chose from different transfer functions.
Layer 2 is the output layer. You can collect result value from this layer.
Also this layer summarize weighted hidden values.
You can chose transfer function from different model.
Transfer functions provided in this version:
- NodeLin  (linear function; fixed for layer 0)
- NodeSigm (sigmoid function = 1/(1+exp(-x))
- NodeTanh (hyperbolic tangent = (exp(x)-exp(-x))/(exp(x)+exp(-x))
- NodeReLU (Rectified LinearUnit = 0.1*x if x<0 else just x (actually Leaky ReLU)
- NodeSmax (exponential normalization for classifing output = exp(xi)/sum(exp(xn))

Neural Network can be defined with random weight for training phase or
can be defined with correct value for use it.
- Random definition:   NNet(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type);
           or          NNet(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type,bool bias,bool mem);
                       (this extended definition allows the use of bias and/or context memory)

- complete definition: NNet(char netdef[]); (by definition string)
                       NNet(const char* filename);  (by file; not for Arduino)
                       save(const char* filename);  (save on file; not for Arduino)

Examples:
Definition of a random net of 2 input nodes, 2 hidden NodeTanh and one NodeLin for output layer
   NNet net(2,2,"NodeTanh",1,"NodeLin");

Creation of similar neural network but with exact weight values
   NNet(char netdef[]);

   where netdef string is defined as:
   char* netdef=
    "L0 2 "                      //layer 0 with 2 nodes
    "L1 2 NodeTanh "             //layer 1 with 2 nodes NodeTanh
opt."HLBS 0.55 -0.1 "            //bias of hidden layer nodes (if present)
    "HLW0 -2.3404 -2.3427 "      //Hidden_Layer_Weigth_ofnode0 value_from_node_0_oflayer0 ...
    "HLW1 0.4820 0.4669 "        //  "       "    "          1    "        "
    "L2 1 NodeLin "              //layer 2 with 1 node NodeLin
opt."OLBS 0.1 "                  //bias of output nodes (if present)
opt."OMW0 0.3202 0.8885 "        //Contex memory to output node 0 from buffer node 0 and 1(if present)
    "OLW0 -2.3558 -3.0901";      //Output_Layer_Weight_ofnode0 value_from_node_0_oflayer1 ...

NB. Always separate token with spaces!

Copyright (C) Daniele Denaro (2019)
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.

********************************************************************************************/

#ifndef NNET_H
#define NNET_H
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RA -0.1
#define RB +0.1
#define LCF 0.01
#define DEC 0.01

#define OptimizerRMSprop true

#define ARDUINONNET    //uncomment if used on ARDUINO or comment if not

#ifdef ARDUINONNET
        typedef struct NNPROGMEMpointers *NNPGM; 
#endif

class NNet
{
    public:
/** Definition of random nnet (nnet to train):
    L0n: number of nodes of layer 0
    L1n: number of nodes of layer 1
    L1Type; name of transfer function used by layer 1
    L2n: number of nodes of layer 2
    L2Type: name of transfer function used by layer 2
    or with bias (for hidden and output layer and/or context memory*/
        NNet(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type);
        NNet(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type,bool bias,bool mem);

/** Creation of a nnet described in netdef string */
        NNet(char netdef[]);

/** Creation of a nnet described in file with name filename (not Arduino)*/
        NNet(const char* filename);

        virtual ~NNet();

/** Basic exploitation function.
    inp array has to contain input values and out array is the buffer that collect result
    (input array size must be >=  NNet lay0n ; output array size must be >= NNet lay2n)*/
        void forw(float inp[],float out[]);

#ifdef ARDUINONNET
/** Second version of forw uses Arduino PROGMEM data (bias and recurrent optional)*/
        void printPROGMEM();
        static NNPGM initNetPROGMEM(void* net,bool fbias,bool fmem);
        static void forwPROGMEM(NNPGM nnp,float inp[],float out[]);
#endif // ARDUINONNET


/** Training function.
    train array is the forced output supplied by user.
    Function returns the squared mean error
    (inp array size must be >= lay0n and train array size must be >= lay2n) */
        float learn(float inp[],float train[]);
/** Learn and propagate error gradient to inp buffer. You can uses these inp
    values to transfer learning process to another NNet in a chain.
    (using baclearn function where bkerr array is the train for this NNet)  */
        float learnpropagate(float inp[],float train[]);
        float backlearn(float inp[],float bkerr[]);

/** Reset optional buffered memory in case of new learning start
    (same functionality as reset capacity in a dinamic system) */
        void resetMem();
/** Reset gradient average just in case of new starting learning process (for RMSprop optimizer) */
        void resetLearnPar();

/** Print nnet structure*/
        void print();

/** Save nnet structure on file*/
        void save(char* filename);
        void savexardu(char* filename);
        void savePROGMEM(char* filename);

/* Utilities */
        void setLearnCoeff(float lcf); //def.: 0.01
        void setRMSpDecay(float dec);  //def.: 0.99 (do not set <0.9)
        void setRMSpOptimizer(bool y); //set RMSprop optimizer (def: true)

        void getHiddenOut(float out[],int dimhid);
        void getNetOut(float out[],int dimout);
        void getWeightsL1fromL0(int nodeL1,float w[],int dimw);
        void getWeightsL2fromL1(int nodeL2,float w[],int dimw);
/* Get L0n or L1n or L2n. layer: 0/1/2 */
        int getnnodes(int layer);

    protected:

    private:
        int lay0n;  //layers dimension
        int lay1n;
        int lay2n;

        struct Wgt  //weights
        {
           float w;
           float Gm2;
        };

        struct Lhid //Hidden node
        {
           float out;
           float err;
           Wgt* wgt;
        };
        struct Lout  //Output node
        {
           float out;
           Wgt* wgt;
        };

        struct Hmem //recursive node (context memory) (optional)
        {
           float hm;
           Wgt* wgt;
        };

        Lhid*  hidn;  //layer hidden pointer
        Lout*  outn;  //layer out pointer

        Hmem*  hmem; //pointer to context memory (hidden buffer) (optional)

        Wgt*  bhid;  //bias for hidden layer (optional)
        Wgt*  bout;  //bias for output layer (optional)

        float (*frw1)(float act);   //pointer to activation function hidden layer
        float (*frw2)(float act);   //pointer to activation function output layer
        float (*learn1)(float out); //pointer to derivative of activation function hidden layer
        float (*learn2)(float out); //pointer to derivative of activation function output layer

        void decodeType1(const char* ty);
        void decodeType2(const char* ty);
        char ty1[12];unsigned char cod1;
        char ty2[12];unsigned char cod2;

        void init(int L0n,int L1n,const char* L1Type,int L2n,const char* L2Type);
        float backp(float inp[],float train[],bool bk);

#ifndef ARDUINONNET
        void outdef(FILE* fp,bool ardu);
        void outdefPROGMEM(FILE* fp);
#endif // ARDUINO

        int readval(char* token);
        int decode(char* token);
        void newline(int stat, char* token);
        void tokenval(char* token);
};

#endif // NNET_H
