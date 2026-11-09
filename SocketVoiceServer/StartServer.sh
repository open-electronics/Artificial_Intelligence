#!/bin/bash 

#******************************************
#Start command for SocketVoiceServer system
#******************************************
#Comment lines 19-20, and uncomment lines 21-22 if you want run python version

#for last OS 
#amixer sset 'Speaker' 80%
#amixer sset 'Mic' 60%

#for previous OS
amixer sset 'Master' 80%
amixer sset 'Capture' 60%

fpath=$(realpath $0)
dname=$(dirname $fpath)
aplay $dname/Utility/Starting.wav
cd $dname/binRasp
./SocketVoiceServer.bin
#cd $dname/Source/
#python SocketVoiceServer.py
aplay $dname/Utility/ServerClosed.wav
