## Face Recognition (too phases: detect and recognition)

This directory contains examples (Python) for face recognition.

Installation of required libraries:

`pip3 install face_recognition` (Python library for face recognition)

Libraries used by examples (but not indispensable):

- Text to speech:  `TextToSpeechLib.py`  
- Speech to text: `SpeechToTextLib.py` 

Model used by face detect phase: `haarcascade_frontalface_default.xml` 

Or `lbpcascade_frontalface_improved.xml`used in TestCameraDetect.py
(but in this case modify its path inside the python file)

Examples:

- `FaceRecon.py`  Program for face (detect and) recognition (it needs a previous saved face image to recognize)
- `VerifyPhotos.py`  This program can choose (interactively) reference photo and choose photo to compare.
- `TestCameraDetect.py`  Program for testing different model for face detecting (HAAR, LBP, HOG, CNN)
  (please modify path to Haar file and Lbp file inside the python program file o put these files into correct path)
