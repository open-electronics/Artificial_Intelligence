#!/bin/bash
#Utility for memory monitoring
#Print memory free amount every second
for i in {1..1000}; do
  read -n 2 -t 1 -s
  if [[ $? -gt 128 ]]
  then 
    fr=0
  else 
    fr=100
  fi    
  riga=$(free -m)
  stringarray=($riga)
  echo ${stringarray[8]} $fr
done
