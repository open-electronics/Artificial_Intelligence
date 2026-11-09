set d=%~p0
set f=%d%\Utility\Starting.wav
powershell -c (New-Object Media.SoundPlayer %f%).PlaySync();
cd ..\SocketVoiceServer\Source
python SocketVoiceServer.py
