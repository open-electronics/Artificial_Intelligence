#ifndef ARDSUMOLIB

#include <Arduino.h>
//**********piedinatura
//motor
#define STBY   4        //pin per stby motori OUT               (STBY)
#define LmVel  5        //pin velocità motore sinistro OUT      (PWMA)
#define RmVel 10        //pin velocità motore destro   OUT      (PWMB)
#define LpinA  7        //pin A direzione motore sinistro OUT   (AIN2)
#define LpinB  6        //pin B direzione motore sinistro OUT   (AIN1)
#define RpinA  8        //pin A direzione motore destro OUT     (BIN1)
#define RpinB  9        //pin B direzione motore destro OUT     (BIN2)
//Ir radar Sharp
#define LirRadar 1     //A1 sensori sharp IR
#define RirRadar 0     //A0 sensori sharp IR
//Ir line sensors
#define LineIrL 13      //D13 ir linea left
#define LineIrC  3      //D3 ir linea center
#define LineIrR  2      //D2 ir linea right
#define LineIrB  2      //A2 ir linea back
//buzzer
#define Buzz  12
//leds;
#define Light 11

void setupPins()
{
   pinMode(LmVel,OUTPUT);
   pinMode(RmVel,OUTPUT);
   pinMode(LpinA,OUTPUT);
   pinMode(LpinB,OUTPUT);
   pinMode(RpinA,OUTPUT);
   pinMode(RpinB,OUTPUT);
   pinMode(STBY,OUTPUT);
   pinMode(Light,OUTPUT);
   pinMode(Buzz,OUTPUT);
   pinMode(LirRadar,INPUT);
   pinMode(RirRadar,INPUT);
   pinMode(LineIrL,INPUT);
   pinMode(LineIrR,INPUT);
   pinMode(LineIrC,INPUT); 
   pinMode(LineIrB,INPUT); 
}

float ReadIrL() //volts% left sensor (from 0:maxdist to 1:mindist)
{
   float val=analogRead(LirRadar);
   val=val/600;
   val=max(0.1,val);
   return val;
}

float ReadIrR() //volts% right sensor (from 0:maxdist to 1:mindist)
{
   float val=analogRead(RirRadar);
   val=val/600;
   val=max(0.1,val);
   return val;
}

void MotorL(float v) //v: from -1 to +1 left motor
{
  int pwr=v*255;
  if (pwr>0) {digitalWrite(LpinA,HIGH);digitalWrite(LpinB,LOW);analogWrite(LmVel,pwr);}  
  if (pwr<0) {digitalWrite(LpinA,LOW);digitalWrite(LpinB,HIGH);analogWrite(LmVel,pwr);}  
  digitalWrite(STBY,HIGH);
  if (pwr==0) {digitalWrite(LpinA,LOW);digitalWrite(LpinB,LOW);analogWrite(LmVel,pwr);}
}

void MotorR(float v) //v: from -1 to +1 right motor
{
  int pwr=v*255;
  if (pwr>0) {digitalWrite(RpinA,HIGH);digitalWrite(RpinB,LOW);analogWrite(RmVel,pwr);}  
  if (pwr<0) {digitalWrite(RpinA,LOW);digitalWrite(RpinB,HIGH);analogWrite(RmVel,pwr);}  
  digitalWrite(STBY,HIGH);
  if (pwr==0) {digitalWrite(RpinA,LOW);digitalWrite(RpinB,LOW);analogWrite(RmVel,pwr);}
}

#endif
