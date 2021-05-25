### Yolo web camera  

Yolo deep learning objects detect using OpenCV as backend.
This program is dedicated to show detected objects captured by camera (yolo model default: yolo4-tiny). OpenCV load network, adapt image from file to frame dimension used for training and detect objects using labels listed in dedicated file.

This script can be used on Window or Jetson, but you have to change comment in these line

`root = "D:\\"    #Windows
#root = "/"        #Linux
prepath = os.path.join(root, "AIData")        # ex. windows
#prepath = os.path.join(root, "home", "jetson", "AIData")     # ex. linux`

That means that configuration files and weights  have to be positioned in:

- D:\\\AIData\\darknet   (Windows)
- /home/jetson/AIData/darknet  (Jetson)

You can change model (network structure). Weights of pre-trained model are defined by a file with similar name.
Labels file is coco.names except for yolov3-tiny-voc model (in this case is voc.names)

Use -h parameter to display help.

`usage: OpencvYoloCamera.py [-h] [--model MODEL] [--labels LABELS] [--thresh THRESH] [--frd FRD FRD] [--camera CAMERA]  [--wdim WDIM] [--nocuda] [-?]`

`OpenCV-YOLO program`

`optional arguments:`
  `-h, --help       show this help message and exit`
  `--model MODEL    net model path (def: yolov4-tiny)`
  `--labels LABELS  path of net object classes (def: coco.names)`
  `--thresh THRESH  confidence threshold (def: 0.3)`
  `--frd FRD FRD    samples frame dimension (def: 416 416)`
  `--camera CAMERA  camera number (def: 0)`
  `--wdim WDIM      win dimension (L:640x480, S:320x240) (def:L)`
  `--nocuda         don't use CUDA (def. use CUDA if present)`
  `-?               Help as -h`

