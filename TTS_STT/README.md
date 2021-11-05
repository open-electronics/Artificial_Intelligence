## Text To Speech and Speech To Text

This directory contains library for TTS and STT.

This library is implemented in Python and can be used on PC, Raspberry Pi or other mini computer that can install Python.

Unfortunately this library needs Internet connection because it uses Google API.  In the future a new directory will be created where off line API library will be described, if a good performance of free software will be evaluated (particularly for Italian language). Anyway latency time is just for computation and API look like local library.

Installation of required libraries:

`pip install SpeechRecognition` (Speech to text)
`pip install PyAudio`  (for microphone) (but before install `pip install wheel` if not present)
for Raspberry Pi `sudo apt-get install python-pyaudio python3-pyaudio`
							`sudo apt-get install flac` (SpeechRecognition needs this coding library)

`pip install gTTS`  (Text to speech)
`pip install miniaudio`

Libraries:

- Text to speech:  `TextToSpeechLib.py`  with functions
  - setLangIt() or setLangEn()  *setting language (It or En)*
  - talk(text)  *where text is a string*
  - talktofile(text, filename)  *create a mp3 file*
- Speech to text: `SpeechToTextLib.py` with functions
  - setLangIt() or setLangEn()
  - listen() *return a string (text recognized); end phrase detected by 0.5 second of silence*
  - startcontinuelisten() *accumulate text detected in a queue; text can be extracted by getText function*
  - getText() *return next text extracted from queue (or None); must be used after previous function*

Four examples of Python scripts are included.

For Speech to text:

`ExampleTranscribe.py`   *Continuous listening and writing every word heard*
`ExampleDialog.py`  *Listen and write (in Italian)*

For Text to speech:

`ExampleSpeech.py`  and `ExampleReadingDoc.py` *(the last one allows you to choose a text document )*

A further example is provided to simulate a vocal translator (It to En):

`ExampleTranslation.py`

But for these uses you need to install this Python library:

`pip install googletrans-3.1.0a0`





